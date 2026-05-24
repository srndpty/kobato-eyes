"""Background indexing tasks used by the tag search tab."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from core.config import PipelineSettings
from core.pipeline import IndexProgress
from ui.viewmodels import TagsViewModel

logger = logging.getLogger(__name__)


class IndexRunnable(QRunnable):
    """Execute the indexing pipeline on a worker thread."""

    class IndexSignals(QObject):
        """Signals emitted by an indexing task."""

        progress = pyqtSignal(int, int, str)
        progressState = pyqtSignal(object)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

    def __init__(
        self,
        view_model: TagsViewModel,
        db_path: Path,
        *,
        settings: PipelineSettings | None = None,
        pre_run: Callable[[], dict[str, object]] | None = None,
        runner: Callable[[Callable[[IndexProgress], None], Callable[[], bool]], dict[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self._view_model = view_model
        self._db_path = Path(db_path)
        self._settings = settings
        self._pre_run = pre_run
        self._runner = runner
        self.signals = self.IndexSignals()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the current indexing run."""

        self._cancel_event.set()

    def _emit_progress(self, progress: IndexProgress) -> None:
        label = progress.phase.name.title()
        if progress.total < 0 and progress.message:
            label = progress.message

        try:
            self.signals.progressState.emit(progress)
            self.signals.progress.emit(progress.done, progress.total, label)
        except RuntimeError:
            return

    def _execute(self) -> dict[str, object]:
        if self._runner is not None:
            return self._runner(self._emit_progress, self._cancel_event.is_set)
        return self._view_model.run_index_once(
            self._db_path,
            settings=self._settings,
            progress_cb=self._emit_progress,
            is_cancelled=self._cancel_event.is_set,
        )

    def run(self) -> None:
        """Run the indexing operation."""

        try:
            extra: dict[str, object] = {}
            if self._pre_run is not None:
                extra = self._pre_run()
            stats = self._execute()
            if extra:
                stats.update(extra)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Indexing failed for database %s", self._db_path)
            self.signals.error.emit(str(exc))
        else:
            self.signals.finished.emit(stats)


__all__ = ["IndexRunnable"]
