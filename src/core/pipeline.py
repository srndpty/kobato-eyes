"""File processing pipeline ensuring watcher events update indices."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional, Set

from PyQt6.QtCore import QObject, QTimer

from core.jobs import BatchJob, JobManager
from core.scanner import DEFAULT_EXTENSIONS
from core.tag_job import TagJobConfig, run_tag_job
from core.watcher import DirectoryWatcher, FileEvent
from db.connection import get_conn
from dup.indexer import DuplicateIndexer, EmbedderProtocol
from index.hnsw import HNSWIndex
from tagger.base import ITagger

logger = logging.getLogger(__name__)


@dataclass
class PipelineSettings:
    roots: list[Path] = field(default_factory=list)
    excluded: list[Path] = field(default_factory=list)
    hamming_threshold: int = 8
    cosine_threshold: float = 0.2
    ssim_threshold: float = 0.9
    model_name: str = "clip-vit"


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


__all__ = ["PipelineSettings", "ProcessingPipeline"]
