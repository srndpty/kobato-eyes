from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from core.db_writer import DBItem
from db.connection import get_conn
from tagger.base import TagCategory, TagResult

from .maintenance import _settle_after_quiesce
from .resolver import _resolve_tagger

# from .loaders import PrefetchLoaderPrepared
from .testhooks import TaggingDeps
from .types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter, _FileRecord

logger = logging.getLogger(__name__)


class TaggingStage:
    def __init__(self, ctx: PipelineContext, emitter: ProgressEmitter, deps: TaggingDeps | None = None):
        self.ctx = ctx
        self.emitter = emitter
        # 実運用ではデフォルト依存を使用、テスト時は差し替え
        self._deps = deps or TaggingDeps()

    def _log_step(self, stage: str, done: int, total: int, last: int) -> int:
        if total <= 0:
            return last
        step = max(1, total // 5)
        if done >= total or done - last >= step:
            logger.info("%s progress: %d/%d", stage, done, total)
            return done
        return last

    def run(self, records: list[_FileRecord]) -> tuple[int, int, list[tuple[int, np.ndarray]]]:
        thresholds = self.ctx.thresholds
        max_tags_map = self.ctx.max_tags_map
        tagger_sig = self.ctx.tagger_sig
        settings = self.ctx.settings
        db_path = self.ctx.db_path

        tag_records = [r for r in records if r.needs_tagging and not r.load_failed]
        self.emitter.emit(IndexProgress(phase=IndexPhase.TAG, done=0, total=len(tag_records)), force=True)
        self.emitter.emit(IndexProgress(phase=IndexPhase.FTS, done=0, total=len(tag_records)), force=True)
        if not tag_records:
            return 0, 0, []

        processed_tags = 0
        fts_processed = 0
        last_logged = 0
        quiesced = False
        dbw = None

        def _dbw_progress(kind: str, done: int, total: int) -> None:
            nonlocal fts_processed
            fts_processed = done
            self.emitter.emit(IndexProgress(phase=IndexPhase.FTS, done=done, total=total))
            try:
                logger.info("finalizing: %s %d/%d", kind.split(".")[0], done, total)
            except Exception:
                pass

        try:
            conn = get_conn(db_path, allow_when_quiesced=True)
            try:
                conn.close()
            except Exception:
                pass

            self._deps.quiesce.begin()
            quiesced = True

            tagger = _resolve_tagger(settings, None, thresholds=thresholds or None, max_tags=max_tags_map or None)
            logger.info("Tagging %d image(s)", len(tag_records))

            dbw = self._deps.dbwriter_factory(
                db_path=db_path,
                flush_chunk=getattr(settings, "db_flush_chunk", 1024),
                fts_topk=getattr(settings, "fts_topk", 128),
                queue_size=int(os.environ.get("KE_DB_QUEUE", "1024")),
                default_tagger_sig=tagger_sig,
                unsafe_fast=True,
                skip_fts=True,
                progress_cb=_dbw_progress,
            )
            dbw.start()
            time.sleep(0.2)
            dbw.raise_if_failed()

            rec_by_path: dict[str, _FileRecord] = {str(r.path): r for r in tag_records}
            tag_paths: list[str] = list(rec_by_path.keys())
            tag_paths.sort(key=lambda p: (Path(p).parent, os.path.getsize(p)))
            current_batch = max(1, int(getattr(settings, "batch_size", 32) or 32))
            prefetch_depth = int(os.environ.get("KE_PREFETCH_DEPTH", "128") or "128") or 128
            io_workers = int(os.environ.get("KE_IO_WORKERS", "12") or "12") or None

            loader = self._deps.loader_factory(
                tag_paths,
                tagger,
                current_batch,
                prefetch_depth,
                io_workers,
            )

            def _infer_with_retry_np(rgb_list: list[np.ndarray]) -> list[TagResult]:
                try:
                    np_batch = tagger.prepare_batch_from_rgb_np(rgb_list)
                    return tagger.infer_batch_prepared(
                        np_batch, thresholds=thresholds or None, max_tags=max_tags_map or None
                    )
                except Exception:
                    if len(rgb_list) <= 1:
                        raise
                    mid = len(rgb_list) // 2
                    return _infer_with_retry_np(rgb_list[:mid]) + _infer_with_retry_np(rgb_list[mid:])

            try:
                loader_iter = iter(loader)
                while True:
                    t_wait0 = time.perf_counter()
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        break

                    wait_batch_ms = (time.perf_counter() - t_wait0) * 1000.0
                    if self.emitter.cancelled(self.ctx.is_cancelled):
                        break
                    if not batch:
                        logger.info("PIPE wait_batch=%.2fms (empty batch)", wait_batch_ms)
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
                        results = _infer_with_retry_np(rgb_list)
                    except Exception:
                        logger.exception("Tagger failed for a prepared batch")
                        continue

                    per_file_items: list[DBItem] = []
                    now_ts = time.time()
                    for rec, result, (w_opt, h_opt) in zip(batch_recs, results, wh_needed):
                        merged: dict[str, tuple[float, TagCategory]] = {}
                        for pred in result.tags:
                            name = pred.name.strip()
                            if not name:
                                continue
                            s = float(pred.score)
                            cur = merged.get(name)
                            if (cur is None) or (s > cur[0]):
                                merged[name] = (s, pred.category)
                        items = [(n, s, int(c)) for n, (s, c) in merged.items()]
                        per_file_items.append(
                            DBItem(rec.file_id, items, w_opt, h_opt, tagger_sig=tagger_sig, tagged_at=now_ts)
                        )

                        # prev_sig = rec.stored_tagger_sig
                        rec.needs_tagging = False
                        rec.tag_exists = True
                        rec.stored_tagger_sig = tagger_sig
                        rec.current_tagger_sig = tagger_sig
                        rec.last_tagged_at = now_ts

                    q_before = None
                    try:
                        q_before = dbw.qsize()
                    except Exception:
                        pass
                    t_enqueue0 = time.perf_counter()
                    for db_item in per_file_items:
                        dbw.put(db_item)
                    enqueue_ms = (time.perf_counter() - t_enqueue0) * 1000.0
                    q_after = None
                    try:
                        q_after = dbw.qsize()
                    except Exception:
                        pass

                    logger.info(
                        "PIPE batch=%d wait_batch=%.2fms enqueue=%.2fms qsize=%s->%s",
                        len(per_file_items),
                        wait_batch_ms,
                        enqueue_ms,
                        q_before,
                        q_after,
                    )
                    processed_tags += len(per_file_items)
                    fts_processed += len(per_file_items)
                    last_logged = self._log_step("Tagging", processed_tags, len(tag_records), last_logged)
                    self.emitter.emit(IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)))
                    self.emitter.emit(IndexProgress(phase=IndexPhase.FTS, done=fts_processed, total=len(tag_records)))

            finally:
                try:
                    loader.close()
                except Exception:
                    pass
                if dbw is not None:
                    try:
                        dbw.stop(flush=True, wait_forever=True)
                    finally:
                        dbw = None
                if quiesced:
                    try:
                        self._deps.quiesce.end()
                    except Exception:
                        logger.exception("end_quiesce failed")
                    quiesced = False
                try:
                    _settle_after_quiesce(str(db_path))
                except Exception:
                    logger.warning("settle_after_quiesce failed; continuing")

            logger.info("Tagging complete: %d image(s) processed", processed_tags)
            self.emitter.emit(
                IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)), force=True
            )
            self.emitter.emit(
                IndexProgress(phase=IndexPhase.FTS, done=fts_processed, total=len(tag_records)), force=True
            )
        except Exception as exc:
            logger.exception("Tagging stage failed: %s", exc)

        return processed_tags, fts_processed, []


__all__ = ["TaggingStage"]
