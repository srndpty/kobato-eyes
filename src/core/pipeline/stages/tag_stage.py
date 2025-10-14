"""Tagging stage responsible for running inference on scanned records."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image

from core.db_writer import DBItem
from core.pipeline.types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter, _FileRecord
from tagger.base import TagCategory

from ..resolver import _resolve_tagger

logger = logging.getLogger(__name__)


class LoaderIterable(Protocol):
    """Iterator interface produced by loader factories."""

    def __iter__(self) -> Iterable[tuple[list[str], np.ndarray, list[tuple[int, int]]]]: ...

    def close(self) -> None: ...


class LoaderFactory(Protocol):
    """Protocol describing loader factories used by the tagging stage."""

    def __call__(
        self,
        paths: list[str],
        tagger,
        batch_size: int,
        prefetch_batches: int,
        io_workers: int | None,
    ) -> LoaderIterable: ...


class TagStageDeps(Protocol):
    """Protocol required by :class:`TagStage`."""

    def loader_factory(
        self,
        paths: list[str],
        tagger,
        batch_size: int,
        prefetch_batches: int,
        io_workers: int | None,
    ) -> LoaderIterable: ...


class _DefaultTagStageDeps:
    """Default dependency provider for :class:`TagStage`."""

    def loader_factory(
        self,
        paths: list[str],
        tagger,
        batch_size: int,
        prefetch_batches: int,
        io_workers: int | None,
    ) -> LoaderIterable:
        from ..loaders import PrefetchLoaderPrepared

        return PrefetchLoaderPrepared(
            paths,
            tagger=tagger,
            batch_size=batch_size,
            prefetch_batches=prefetch_batches,
            io_workers=io_workers,
        )


@dataclass(slots=True)
class TagStageResult:
    """Result object returned by :class:`TagStage`."""

    records: list[_FileRecord]
    db_items: list[DBItem]
    tagged_count: int


class TagStage:
    """Stage that performs model inference for files requiring tagging."""

    def __init__(self, deps: TagStageDeps | None = None) -> None:
        self._deps = deps or _DefaultTagStageDeps()

    @staticmethod
    def _sort_key(path: str) -> tuple[Path, int]:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return Path(path).parent, size

    def _log_step(self, stage: str, done: int, total: int, last: int) -> int:
        if total <= 0:
            return last
        step = max(1, total // 5)
        if done >= total or done - last >= step:
            logger.info("%s progress: %d/%d", stage, done, total)
            return done
        return last

    def run(
        self,
        ctx: PipelineContext,
        emitter: ProgressEmitter,
        records: list[_FileRecord],
    ) -> TagStageResult:
        """Run tagging inference and prepare DB items for persistence."""
        import traceback

        traceback.print_stack()
        logger.info("ctx.thresholds: %s", ctx.thresholds)
        thresholds = ctx.thresholds
        max_tags_map = ctx.max_tags_map
        tagger_sig = ctx.tagger_sig
        settings = ctx.settings

        tag_records = [r for r in records if r.needs_tagging and not r.load_failed]
        emitter.emit(IndexProgress(phase=IndexPhase.TAG, done=0, total=len(tag_records)), force=True)
        if not tag_records:
            return TagStageResult(records=records, db_items=[], tagged_count=0)

        processed_tags = 0
        last_logged = 0
        db_items: list[DBItem] = []

        tagger, th_fallback, k_fallback = _resolve_tagger(
            settings,
            ctx.tagger_override,
            thresholds=thresholds or None,
            max_tags=max_tags_map or None,
        )
        # 呼び出し側が未指定だった場合のみ、resolver 側のデフォルトを反映
        if not thresholds and th_fallback:
            thresholds = th_fallback
        if (not max_tags_map) and k_fallback:
            max_tags_map = k_fallback
        logger.info("Tagging %d image(s)", len(tag_records))

        rec_by_path: dict[str, _FileRecord] = {str(r.path): r for r in tag_records}
        tag_paths: list[str] = list(rec_by_path.keys())
        tag_paths.sort(key=self._sort_key)

        configured_batch = getattr(settings, "batch_size", 32)
        try:
            current_batch = int(configured_batch)
        except (TypeError, ValueError):
            logger.warning("Invalid batch size configuration %r; falling back to default", configured_batch)
            current_batch = 32
        if current_batch < 1:
            logger.warning(
                "Configured batch size %d is below 1; using minimum batch size of 1",
                current_batch,
            )
            current_batch = 1
        prefetch_depth = int(os.environ.get("KE_PREFETCH_DEPTH", "128") or "128") or 128
        io_workers = int(os.environ.get("KE_IO_WORKERS", "12") or "12") or None

        loader = self._deps.loader_factory(tag_paths, tagger, current_batch, prefetch_depth, io_workers)

        supports_prepared = callable(getattr(tagger, "infer_batch_prepared", None))

        def _infer_prepared_with_retry(np_batch: np.ndarray):
            if not supports_prepared:
                raise AttributeError("Tagger does not implement infer_batch_prepared")
            try:
                return tagger.infer_batch_prepared(
                    np_batch, thresholds=thresholds or None, max_tags=max_tags_map or None
                )
            except Exception:
                n = np_batch.shape[0]
                if n <= 1:
                    raise
                mid = n // 2
                left = _infer_prepared_with_retry(np_batch[:mid])
                right = _infer_prepared_with_retry(np_batch[mid:])
                return left + right

        try:
            loader_iter = iter(loader)
            while True:
                # すでに全件処理済みなら明示的に抜ける（安全装置）
                if processed_tags >= len(tag_records):
                    logger.info("Processed all %d images; stopping loader loop explicitly.", len(tag_records))
                    break
                t_wait0 = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break

                wait_batch_ms = (time.perf_counter() - t_wait0) * 1000.0
                if emitter.cancelled(ctx.is_cancelled):
                    break
                if not batch:
                    logger.info("PIPE wait_batch=%.2fms (empty batch)", wait_batch_ms)
                    time.sleep(0.05)
                    continue
                batch_paths, batch_np_rgb, sizes = batch

                batch_recs: list[_FileRecord] = []
                rgb_list: list[np.ndarray] = []
                wh_needed: list[tuple[int | None, int | None]] = []
                for p, arr, (_w, _h) in zip(batch_paths, batch_np_rgb, sizes):
                    if arr is None:
                        continue
                    rec = rec_by_path.get(p)
                    if rec is None:
                        continue
                    batch_recs.append(rec)
                    rgb_list.append(arr)
                    need_wh = rec.is_new or rec.changed or rec.width is None or rec.height is None
                    if need_wh:
                        h, w = arr.shape[:2]
                        wh_needed.append((int(w), int(h)))
                    else:
                        wh_needed.append((None, None))
                if not rgb_list:
                    continue

                qsz = getattr(loader, "qsize", lambda: -1)()
                logger.info("PIPE batch=%d wait_batch=%.2fms loader_qsize=%d", len(rgb_list), wait_batch_ms, qsz)

                try:
                    if supports_prepared:
                        results = _infer_prepared_with_retry(batch_np_rgb)
                    else:
                        pil_batch = [Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8)) for arr in rgb_list]
                        results = tagger.infer_batch(
                            pil_batch, thresholds=thresholds or None, max_tags=max_tags_map or None
                        )
                except Exception:
                    logger.exception(
                        "Tagger failed for a prepared batch"
                        if supports_prepared
                        else "Tagger fallback inference failed"
                    )
                    continue

                now_ts = time.time()
                for rec, result, (w_opt, h_opt) in zip(batch_recs, results, wh_needed):
                    merged: dict[str, tuple[float, TagCategory]] = {}
                    for pred in result.tags:
                        name = pred.name.strip()
                        if not name:
                            continue
                        score = float(pred.score)
                        current = merged.get(name)
                        if (current is None) or (score > current[0]):
                            merged[name] = (score, pred.category)
                    items = [(name, score, int(category)) for name, (score, category) in merged.items()]
                    db_items.append(
                        DBItem(
                            rec.file_id,
                            items,
                            w_opt,
                            h_opt,
                            tagger_sig=tagger_sig,
                            tagged_at=now_ts,
                        )
                    )
                    rec.needs_tagging = False
                    rec.tag_exists = True
                    rec.stored_tagger_sig = tagger_sig
                    rec.current_tagger_sig = tagger_sig
                    rec.last_tagged_at = now_ts

                processed_tags += len(batch_recs)
                last_logged = self._log_step("Tagging", processed_tags, len(tag_records), last_logged)
                emitter.emit(IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)))

        finally:
            try:
                loader.close()
            except Exception:
                pass

        logger.info("Tagging complete: %d image(s) processed", processed_tags)
        emitter.emit(
            IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)),
            force=True,
        )
        return TagStageResult(records=records, db_items=db_items, tagged_count=processed_tags)


__all__ = ["TagStage", "TagStageResult", "TagStageDeps"]
