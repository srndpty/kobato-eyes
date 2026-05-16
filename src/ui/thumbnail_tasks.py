"""Background thumbnail tasks for tag search results."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtGui import QPixmap

from utils.image_io import get_thumbnail


class ThumbnailSignal(QObject):
    """Signal object shared by thumbnail workers."""

    finished = pyqtSignal(int, QPixmap)


class ThumbnailTask(QRunnable):
    """Load one thumbnail in a worker thread and emit it to the UI."""

    def __init__(self, row: int, path: Path, width: int, height: int, signal: ThumbnailSignal) -> None:
        super().__init__()
        self._row = row
        self._path = path
        self._width = width
        self._height = height
        self._signal = signal

    def run(self) -> None:
        """Run the thumbnail load."""

        pixmap = get_thumbnail(self._path, self._width, self._height)
        self._signal.finished.emit(self._row, pixmap)


__all__ = ["ThumbnailSignal", "ThumbnailTask"]
