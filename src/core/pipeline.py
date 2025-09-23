"""File processing pipeline ensuring watcher events update indices."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional, Set

import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, QTimer

from core.config import load_settings
from core.jobs import BatchJob, JobManager
from core.scanner import DEFAULT_EXTENSIONS, iter_images
from core.settings import PipelineSettings
from core.tag_job import TagJobConfig, run_tag_job
from core.watcher import DirectoryWatcher, FileEvent
from db.connection import get_conn
from db.repository import (
    get_file_by_path,
    mark_indexed_at,
    replace_file_tags,
    update_fts,
    upsert_embedding,
    upsert_file,
    upsert_signatures,
    upsert_tags,
)
from dup.indexer import DuplicateIndexer, EmbedderProtocol, add_embeddings_to_hnsw, load_hnsw_index, save_hnsw_index
from index.hnsw import HNSWIndex
from sig.phash import dhash, phash
from tagger.base import ITagger, TagCategory
from utils.hash import compute_sha256
from utils.image_io import safe_load_image

logger = logging.getLogger(__name__)


class _FileProcessJob(BatchJob):
    def __init__(
        self,
        paths: Iterable[Path],
        *,
        db_path: Path,
        tagger: ITagger,
        embedder: EmbedderProtocol,
        hnsw_index: HNSWIndex,
        model_name: str,
        tag_config: TagJobConfig,
    ) -> None:
        super().__init__(list(paths))
        self._db_path = db_path
        self._tagger = tagger
        self._embedder = embedder
        self._index = hnsw_index
        self._model_name = model_name
        self._tag_config = tag_config
        self._conn = None
        self._indexer: DuplicateIndexer | None = None

    def prepare(self) -> None:
        self._conn = get_conn(self._db_path)
        self._indexer = DuplicateIndexer(
            self._conn,
            self._embedder,
            self._index,
            model_name=self._model_name,
        )
        logger.debug("Prepared processing job for %d files", len(self.items))

    def load_item(self, item: Path) -> Path:
        return item

    def process_item(self, item: Path, loaded: Path) -> tuple[Path, Optional[int]] | None:
        if self._conn is None or self._indexer is None:
            logger.error("Job not prepared before processing")
            return None
        if not loaded.exists():
            logger.info("Skipping missing file: %s", loaded)
            return None
        try:
            output = run_tag_job(self._tagger, loaded, self._conn, config=self._tag_config)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Tagging failed for %s", loaded)
            return None
        file_id = output.file_id if output else None
        try:
            indexed_ids = self._indexer.index_paths([loaded])
            if indexed_ids and file_id is None:
                file_id = indexed_ids[0]
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Indexing failed for %s", loaded)
        return (loaded, file_id)

    def write_item(self, item: Path, processed: tuple[Path, Optional[int]] | None) -> None:
        if processed is None:
            return
        _, file_id = processed
        if file_id is None:
            logger.warning("No file ID resolved after processing %s", item)
        else:
            logger.debug("Processed file %s (id=%s)", item, file_id)

    def cleanup(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        logger.debug("Processing job finished")


class ProcessingPipeline(QObject):
    """Drive watcher events through tagging and duplicate indexing."""

    def __init__(
        self,
        *,
        db_path: Path,
        tagger: ITagger,
        embedder: EmbedderProtocol,
        hnsw_index: HNSWIndex,
        job_manager: JobManager,
        settings: PipelineSettings | None = None,
    ) -> None:
        super().__init__()
        self._db_path = Path(db_path)
        self._tagger = tagger
        self._embedder = embedder
        self._index = hnsw_index
        self._job_manager = job_manager
        self._settings = settings or PipelineSettings()
        self._queue: Queue[FileEvent] = Queue()
        self._watcher: DirectoryWatcher | None = None
        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(250)
        self._drain_timer.timeout.connect(self._drain_events)
        self._scheduled: Set[Path] = set()
        self._tag_config = TagJobConfig()

    def update_settings(self, settings: PipelineSettings) -> None:
        logger.info("Updating pipeline settings: %s", settings)
        self._settings = settings
        if self._watcher is not None:
            self.stop()
            self.start()

    def start(self) -> None:
        if self._watcher is not None:
            return
        if not self._settings.roots:
            logger.warning("Cannot start pipeline without roots")
            return
        self._watcher = DirectoryWatcher(
            self._settings.roots,
            self._queue,
            excluded=self._settings.excluded,
            extensions=DEFAULT_EXTENSIONS,
        )
        self._watcher.start()
        self._drain_timer.start()
        logger.info("Processing pipeline started")

    def stop(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        self._drain_timer.stop()
        self._scheduled.clear()
        logger.info("Processing pipeline stopped")

    def shutdown(self) -> None:
        self.stop()
        self._job_manager.shutdown()

    def enqueue_path(self, path: Path) -> None:
        self._schedule_path(path)

    def _drain_events(self) -> None:
        while not self._queue.empty():
            event = self._queue.get()
            path = Path(event.path)
            self._schedule_path(path)

    def _schedule_path(self, path: Path) -> None:
        if not path.suffix:
            return
        if path.suffix.lower() not in DEFAULT_EXTENSIONS:
            return
        resolved = path.resolve()
        if resolved in self._scheduled:
            return
        self._scheduled.add(resolved)
        logger.debug("Scheduling processing for %s", resolved)
        job = _FileProcessJob(
            [resolved],
            db_path=self._db_path,
            tagger=self._tagger,
            embedder=self._embedder,
            hnsw_index=self._index,
            model_name=self._settings.model_name,
            tag_config=self._tag_config,
        )
        signals = self._job_manager.submit(job)
        signals.finished.connect(lambda path=resolved: self._scheduled.discard(path))


@dataclass
class _FileRecord:
    """Metadata captured for a file during a one-shot indexing run."""

    file_id: int
    path: Path
    size: int
    mtime: float
    sha: str
    is_new: bool
    changed: bool
    tag_exists: bool
    embed_exists: bool
    needs_tagging: bool
    needs_embedding: bool
    image: Image.Image | None = None
    width: int | None = None
    height: int | None = None
    load_failed: bool = False


def run_index_once(
    db_path: str | Path,
    settings: PipelineSettings | None = None,
    *,
    tagger_override: ITagger | None = None,
    embedder_override: EmbedderProtocol | None = None,
) -> dict[str, object]:
    """Perform a full indexing pass across all configured roots."""
    start_time = time.perf_counter()
    settings = settings or load_settings()
    stats: dict[str, object] = {
        "scanned": 0,
        "new_or_changed": 0,
        "tagged": 0,
        "signatures": 0,
        "embedded": 0,
        "hnsw_added": 0,
        "elapsed_sec": 0.0,
    }

    roots = [Path(root).expanduser() for root in settings.roots if root]
    roots = [root for root in roots if root.exists()]
    if not roots:
        logger.info("No valid roots configured; skipping indexing run.")
        stats["elapsed_sec"] = time.perf_counter() - start_time
        return stats

    excluded_paths = [Path(path).expanduser() for path in settings.excludes if path]
    allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}
    batch_size = max(1, settings.batch_size)

    records: list[_FileRecord] = []
    hnsw_additions: list[tuple[int, np.ndarray]] = []

    conn = get_conn(db_path)
    try:
        logger.info("Scanning %d root(s) for eligible images", len(roots))
        for image_path in iter_images(roots, excluded=excluded_paths, extensions=allow_exts):
            stats["scanned"] += 1
            try:
                stat = image_path.stat()
            except OSError as exc:
                logger.warning("Failed to stat %s: %s", image_path, exc)
                continue
            try:
                sha = compute_sha256(image_path)
            except OSError as exc:
                logger.warning("Failed to hash %s: %s", image_path, exc)
                continue

            row = get_file_by_path(conn, str(image_path))
            is_new = row is None
            changed = True
            if row is not None:
                changed = (
                    row["sha256"] != sha
                    or float(row["mtime"] or 0.0) != stat.st_mtime
                    or int(row["size"] or 0) != stat.st_size
                )

            indexed_at = None if changed else (row["indexed_at"] if row else None)
            file_id = upsert_file(
                conn,
                path=str(image_path),
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha256=sha,
                indexed_at=indexed_at,
            )

            tag_exists = (
                conn.execute(
                    "SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1",
                    (file_id,),
                ).fetchone()
                is not None
            )
            embed_exists = (
                conn.execute(
                    "SELECT 1 FROM embeddings WHERE file_id = ? AND model = ? LIMIT 1",
                    (file_id, settings.embed_model.name),
                ).fetchone()
                is not None
            )

            record = _FileRecord(
                file_id=file_id,
                path=image_path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha=sha,
                is_new=is_new,
                changed=changed,
                tag_exists=tag_exists,
                embed_exists=embed_exists,
                needs_tagging=is_new or changed or not tag_exists,
                needs_embedding=is_new or changed or not embed_exists,
            )
            records.append(record)
        conn.commit()

        stats["new_or_changed"] = sum(1 for record in records if record.is_new or record.changed)
        logger.info(
            "Scan complete: %d file(s) seen, %d new or changed",
            stats["scanned"],
            stats["new_or_changed"],
        )

        def ensure_image_loaded(record: _FileRecord) -> bool:
            if record.load_failed:
                return False
            if record.image is None:
                image = safe_load_image(record.path)
                if image is None:
                    record.load_failed = True
                    logger.warning("Skipping unreadable image %s", record.path)
                    return False
                record.image = image
                record.width = image.width
                record.height = image.height
            return True

        def log_progress(stage: str, processed: int, total: int, last_value: int) -> int:
            if total <= 0:
                return last_value
            step = max(1, total // 5)
            if processed >= total or processed - last_value >= step:
                logger.info("%s progress: %d/%d", stage, processed, total)
                return processed
            return last_value

        tag_records = [record for record in records if record.needs_tagging]
        if tag_records:
            try:
                tagger = _resolve_tagger(settings, tagger_override)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to instantiate tagger: %s", exc)
            else:
                thresholds = _build_threshold_map(settings.tagger.thresholds)
                logger.info("Tagging %d image(s)", len(tag_records))
                processed_tags = 0
                last_logged = 0
                idx = 0
                current_batch = batch_size
                while idx < len(tag_records):
                    current_batch = max(1, current_batch)
                    batch_slice = tag_records[idx : idx + current_batch]
                    images: list[Image.Image] = []
                    valid_records: list[_FileRecord] = []
                    for record in batch_slice:
                        if ensure_image_loaded(record) and record.image is not None:
                            images.append(record.image)
                            valid_records.append(record)
                    if not images:
                        idx += len(batch_slice)
                        continue
                    try:
                        results = tagger.infer_batch(images, thresholds=thresholds)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        if current_batch > 1:
                            current_batch = max(1, current_batch // 2)
                            logger.warning(
                                "Tagger batch failed (%s); reducing batch size to %d",
                                exc,
                                current_batch,
                            )
                            continue
                        logger.exception("Tagger failed for batch starting with %s", valid_records[0].path)
                        idx += len(batch_slice)
                        continue
                    for record, image, result in zip(valid_records, images, results):
                        merged: dict[str, float] = {}
                        categories: dict[str, TagCategory] = {}
                        for prediction in result.tags:
                            name = prediction.name.strip()
                            if not name:
                                continue
                            score = float(prediction.score)
                            existing = merged.get(name)
                            if existing is None or score > existing:
                                merged[name] = score
                                categories[name] = prediction.category
                        if merged:
                            tag_defs = [{"name": name, "category": int(categories[name])} for name in merged]
                            tag_id_map = upsert_tags(conn, tag_defs)
                            tag_scores = [
                                (tag_id_map[name], merged[name])
                                for name in sorted(merged, key=merged.get, reverse=True)
                            ]
                            replace_file_tags(conn, record.file_id, tag_scores)
                            update_fts(conn, record.file_id, " ".join(merged.keys()))
                        else:
                            replace_file_tags(conn, record.file_id, [])
                            update_fts(conn, record.file_id, None)
                        upsert_file(
                            conn,
                            path=str(record.path),
                            size=record.size,
                            mtime=record.mtime,
                            sha256=record.sha,
                            width=image.width,
                            height=image.height,
                        )
                        record.needs_tagging = False
                        record.tag_exists = True
                        processed_tags += 1
                        last_logged = log_progress(
                            "Tagging",
                            processed_tags,
                            len(tag_records),
                            last_logged,
                        )
                    idx += len(batch_slice)
                conn.commit()
                stats["tagged"] = processed_tags
                logger.info("Tagging complete: %d image(s) processed", processed_tags)

        embed_records = [record for record in records if record.needs_embedding and not record.load_failed]
        if embed_records:
            try:
                embedder = embedder_override or _resolve_embedder(settings)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to instantiate embedder: %s", exc)
            else:
                logger.info("Embedding %d image(s)", len(embed_records))
                processed_embeddings = 0
                last_logged = 0
                idx = 0
                current_batch = batch_size
                while idx < len(embed_records):
                    current_batch = max(1, current_batch)
                    batch_slice = embed_records[idx : idx + current_batch]
                    images: list[Image.Image] = []
                    valid_records: list[_FileRecord] = []
                    for record in batch_slice:
                        if ensure_image_loaded(record) and record.image is not None:
                            images.append(record.image)
                            valid_records.append(record)
                    if not images:
                        idx += len(batch_slice)
                        continue
                    try:
                        vectors = embedder.embed_images(images)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        if current_batch > 1:
                            current_batch = max(1, current_batch // 2)
                            logger.warning(
                                "Embedding batch failed (%s); reducing batch size to %d",
                                exc,
                                current_batch,
                            )
                            continue
                        logger.exception("Embedding failed for batch starting with %s", valid_records[0].path)
                        idx += len(batch_slice)
                        continue
                    if len(vectors) != len(valid_records):
                        logger.error(
                            "Embedder returned %d vectors for %d images; skipping batch",
                            len(vectors),
                            len(valid_records),
                        )
                        idx += len(batch_slice)
                        continue
                    for record, image, vector in zip(valid_records, images, vectors):
                        array = np.asarray(vector, dtype=np.float32)
                        if array.ndim != 1:
                            array = np.reshape(array, (-1,))
                        if not array.flags.c_contiguous:
                            array = np.ascontiguousarray(array, dtype=np.float32)
                        else:
                            array = array.astype(np.float32, copy=True)
                        norm = float(np.linalg.norm(array))
                        if norm > 0:
                            array /= norm
                        upsert_file(
                            conn,
                            path=str(record.path),
                            size=record.size,
                            mtime=record.mtime,
                            sha256=record.sha,
                            width=image.width,
                            height=image.height,
                        )
                        upsert_signatures(
                            conn,
                            file_id=record.file_id,
                            phash_u64=phash(image),
                            dhash_u64=dhash(image),
                        )
                        upsert_embedding(
                            conn,
                            file_id=record.file_id,
                            model=settings.embed_model.name,
                            dim=array.shape[0],
                            vector=array.tobytes(),
                        )
                        was_missing = not record.embed_exists
                        mark_indexed_at(conn, record.file_id, indexed_at=time.time())
                        record.needs_embedding = False
                        record.embed_exists = True
                        processed_embeddings += 1
                        if record.is_new or was_missing:
                            hnsw_additions.append((record.file_id, array.copy()))
                        last_logged = log_progress(
                            "Embedding",
                            processed_embeddings,
                            len(embed_records),
                            last_logged,
                        )
                    idx += len(batch_slice)
                conn.commit()
                stats["signatures"] = processed_embeddings
                stats["embedded"] = processed_embeddings
                logger.info("Embedding complete: %d image(s) processed", processed_embeddings)

        for record in records:
            record.image = None

    finally:
        conn.close()

    if hnsw_additions:
        dim = hnsw_additions[0][1].shape[0]
        index_dir = Path(settings.index_dir).expanduser()
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "hnsw_cosine.bin"
        try:
            index = load_hnsw_index(index_path, dim=dim)
            added = add_embeddings_to_hnsw(index, hnsw_additions, dim=dim)
            if added:
                save_hnsw_index(index, index_path)
            stats["hnsw_added"] = added
            logger.info("HNSW index updated with %d new vector(s)", added)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to update HNSW index: %s", exc)

    stats["elapsed_sec"] = time.perf_counter() - start_time
    logger.info(
        "Indexing complete: scanned=%d, new=%d, tagged=%d, embedded=%d, hnsw_added=%d (%.2fs)",
        stats["scanned"],
        stats["new_or_changed"],
        stats["tagged"],
        stats["embedded"],
        stats["hnsw_added"],
        stats["elapsed_sec"],
    )
    return stats


def _resolve_tagger(settings: PipelineSettings, override: ITagger | None) -> ITagger:
    if override is not None:
        return override
    name = settings.tagger.name.lower()
    if name == "dummy":
        from tagger.dummy import DummyTagger

        return DummyTagger()
    if name == "wd14-onnx":
        from tagger.wd14_onnx import WD14Tagger

        if not settings.tagger.model_path:
            raise ValueError("WD14 tagger requires a model_path setting")
        model_path = Path(settings.tagger.model_path)
        labels_csv = model_path.with_suffix(".csv")
        return WD14Tagger(model_path, labels_csv)
    raise ValueError(f"Unknown tagger '{settings.tagger.name}'")


def _resolve_embedder(settings: PipelineSettings) -> EmbedderProtocol:
    from sig.embedder import OpenClipEmbedder

    return OpenClipEmbedder(
        settings.embed_model.name,
        settings.embed_model.pretrained,
        device=settings.embed_model.device,
        batch_size=settings.batch_size,
    )


def _build_threshold_map(thresholds: dict[str, float]) -> dict[TagCategory, float]:
    mapping: dict[TagCategory, float] = {}
    for key, value in thresholds.items():
        category = {
            "general": TagCategory.GENERAL,
            "character": TagCategory.CHARACTER,
            "copyright": TagCategory.COPYRIGHT,
            "artist": TagCategory.ARTIST,
            "meta": TagCategory.META,
            "rating": TagCategory.RATING,
        }.get(key.lower())
        if category is not None:
            mapping[category] = float(value)
    return mapping


__all__ = ["PipelineSettings", "ProcessingPipeline", "run_index_once"]
