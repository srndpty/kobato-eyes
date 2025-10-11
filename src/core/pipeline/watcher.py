from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence, Set

from core.jobs import BatchJob, JobManager
from core.scanner import DEFAULT_EXTENSIONS
from core.config import PipelineSettings
from core.tag_job import TagJobConfig, run_tag_job
from db.connection import bootstrap_if_needed, get_conn
from tagger.base import ITagger
from utils.env import is_headless
from utils.paths import ensure_dirs

from .signature import _build_max_tags_map, _build_threshold_map, current_tagger_sig

if is_headless():

    class QObject:  # type: ignore[too-many-ancestors]
        def __init__(self, *args, **kwargs) -> None:
            pass

        def deleteLater(self) -> None:
            pass
else:  # pragma: no cover
    from PyQt6.QtCore import QObject

logger = logging.getLogger(__name__)


class _Indexer(Protocol):
    """Protocol describing the minimal interface for indexing paths."""

    def index_paths(self, paths: Iterable[Path]) -> Sequence[int]:
        """Index ``paths`` and return the indexed database identifiers."""


class _FileProcessJob(BatchJob):
    def __init__(
        self,
        paths: Iterable[Path],
        *,
        db_path: Path,
        tagger: ITagger,
        tag_config: TagJobConfig,
        indexer: _Indexer | None,
    ) -> None:
        super().__init__(list(paths))
        self._db_path = db_path
        self._tagger = tagger
        self._tag_config = tag_config
        self._indexer = indexer
        self._conn = None

    def prepare(self) -> None:
        self._conn = get_conn(self._db_path)
        logger.debug("Prepared processing job for %d files", len(self.items))

    def load_item(self, item: Path) -> Path:
        return item

    def process_item(self, item: Path, loaded: Path) -> tuple[Path, Optional[int]] | None:
        if self._conn is None:
            logger.error("Job not prepared before processing")
            return None
        if not loaded.exists():
            logger.info("Skipping missing file: %s", loaded)
            return None
        try:
            output = run_tag_job(self._tagger, loaded, self._conn, config=self._tag_config)
        except Exception:
            logger.exception("Tagging failed for %s", loaded)
            return None
        file_id = output.file_id if output else None
        if self._indexer is None:
            logger.debug("Indexer not configured; skipping duplicate indexing for %s", loaded)
            return (loaded, file_id)
        try:
            indexed_ids = self._indexer.index_paths([loaded])
            if indexed_ids and file_id is None:
                file_id = indexed_ids[0]
        except Exception:
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
        job_manager: JobManager,
        indexer: _Indexer | None = None,
        settings: PipelineSettings | None = None,
    ) -> None:
        super().__init__()
        resolved_db = Path(db_path).expanduser()
        db_literal = str(resolved_db)
        if not db_literal.startswith("file:") and db_literal != ":memory:":
            ensure_dirs()
            resolved_db.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved_db
        self._tagger = tagger
        self._job_manager = job_manager
        self._indexer = indexer
        self._settings = settings or PipelineSettings()
        self._scheduled: Set[Path] = set()
        self._tag_config = TagJobConfig()
        self._tagger_sig: str | None = None
        self._refresh_tag_config()

    def _refresh_tag_config(self) -> None:
        thresholds = _build_threshold_map(self._settings.tagger.thresholds)
        max_tags_map = _build_max_tags_map(getattr(self._settings.tagger, "max_tags", None))
        self._tagger_sig = current_tagger_sig(
            self._settings,
            thresholds=thresholds,
            max_tags=max_tags_map,
        )
        self._tag_config = TagJobConfig(
            thresholds=thresholds or None,
            max_tags=max_tags_map or None,
            tagger_sig=self._tagger_sig,
        )

    def update_settings(self, settings: PipelineSettings) -> None:
        logger.info("Updating pipeline settings: %s", settings)
        self._settings = settings
        self._refresh_tag_config()

    def stop(self) -> None:
        self._scheduled.clear()
        logger.info("Processing pipeline stopped")

    def shutdown(self) -> None:
        self.stop()
        self._job_manager.shutdown()

    def enqueue_path(self, path: Path) -> None:
        self.enqueue_index([path])

    def enqueue_index(
        self,
        paths: Iterable[Path],
        *,
        indexer: _Indexer | None = None,
    ) -> None:
        bootstrap_if_needed(self._db_path)
        allow_exts = {ext.lower() for ext in (self._settings.allow_exts or DEFAULT_EXTENSIONS)}
        scheduled_now: set[Path] = set()
        resolved_paths: list[Path] = []
        for path in paths:
            candidate = Path(path)
            suffix = candidate.suffix.lower()
            if not suffix or suffix not in allow_exts:
                continue
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                resolved = candidate.expanduser().absolute()
            if resolved in self._scheduled or resolved in scheduled_now:
                continue
            scheduled_now.add(resolved)
            resolved_paths.append(resolved)
        if not resolved_paths:
            return
        self._scheduled.update(scheduled_now)
        logger.debug("Scheduling processing for %d file(s)", len(resolved_paths))
        active_indexer = indexer if indexer is not None else self._indexer
        job = _FileProcessJob(
            resolved_paths,
            db_path=self._db_path,
            tagger=self._tagger,
            tag_config=self._tag_config,
            indexer=active_indexer,
        )
        signals = self._job_manager.submit(job)

        def _on_finished() -> None:
            for resolved in resolved_paths:
                self._scheduled.discard(resolved)

        signals.finished.connect(_on_finished)
