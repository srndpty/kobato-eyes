"""Background thumbnail tasks for tag search results."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtGui import QPixmap

from utils.image_io import get_thumbnail

logger = logging.getLogger(__name__)


class ThumbnailSignal(QObject):
    """Signal object shared by thumbnail workers."""

    finished = pyqtSignal(int, int, QPixmap)


class ThumbnailTask(QRunnable):
    """Load one thumbnail in a worker thread and emit it to the UI."""

    def __init__(self, row: int, file_id: int, path: Path, width: int, height: int, signal: ThumbnailSignal) -> None:
        super().__init__()
        self._row = row
        self._file_id = int(file_id)
        self._path = path
        self._width = width
        self._height = height
        self._signal = signal
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation before the thumbnail is emitted."""

        self._cancelled = True

    def run(self) -> None:
        """Run the thumbnail load."""

        if self._cancelled:
            return
        try:
            pixmap = get_thumbnail(self._path, self._width, self._height)
        except Exception as exc:
            logger.warning("ThumbnailTask: failed to load %s: %s", self._path, exc)
            pixmap = QPixmap()
        if self._cancelled:
            return
        self._signal.finished.emit(self._row, self._file_id, pixmap)


__all__ = ["ThumbnailSignal", "ThumbnailTask"]
