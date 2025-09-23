"""UI for tag-based search in kobato-eyes."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from PyQt6.QtCore import QModelIndex, QObject, QRunnable, QSize, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QPixmap, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.query import translate_query
from db.connection import get_conn
from db.repository import search_files
from utils.image_io import get_thumbnail


class _ThumbnailSignal(QObject):
    finished = pyqtSignal(int, QPixmap)


class _ThumbnailTask(QRunnable):
    def __init__(
        self,
        row: int,
        path: Path,
        width: int,
        height: int,
        signal: _ThumbnailSignal,
    ) -> None:
        super().__init__()
        self._row = row
        self._path = path
        self._width = width
        self._height = height
        self._signal = signal

    def run(self) -> None:  # noqa: D401 - QRunnable contract
        pixmap = get_thumbnail(self._path, self._width, self._height)
        self._signal.finished.emit(self._row, pixmap)


class TagsTab(QWidget):
    """Provide a search bar and tabular results for tag queries."""

    _PAGE_SIZE = 200
    _THUMB_SIZE = 128

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._query_input = QLineEdit(self)
        self._query_input.setPlaceholderText("Search tags…")
        self._search_button = QPushButton("Search", self)
        self._load_more_button = QPushButton("Load more", self)
        self._load_more_button.setEnabled(False)
        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)

        self._results_view = QTableView(self)
        self._results_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._results_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._results_view.doubleClicked.connect(self._on_row_double_clicked)
        self._results_view.setIconSize(QSize(self._THUMB_SIZE, self._THUMB_SIZE))

        headers = [
            "Thumb",
            "File name",
            "Folder",
            "Size",
            "Dim",
            "Modified",
            "Top 5 Tags",
        ]
        self._model = QStandardItemModel(0, len(headers), self)
        self._model.setHorizontalHeaderLabels(headers)
        self._results_view.setModel(self._model)
        self._results_view.horizontalHeader().setStretchLastSection(True)

        search_layout = QHBoxLayout()
        search_layout.addWidget(self._query_input)
        search_layout.addWidget(self._search_button)

        layout = QVBoxLayout(self)
        layout.addLayout(search_layout)
        layout.addWidget(self._status_label)
        layout.addWidget(self._results_view)
        layout.addWidget(self._load_more_button)

        self._search_button.clicked.connect(self._on_search_clicked)
        self._load_more_button.clicked.connect(self._on_load_more_clicked)
        self._query_input.returnPressed.connect(self._on_search_clicked)

        self._conn = get_conn("kobato-eyes.db")
        self.destroyed.connect(lambda: self._conn.close())

        self._current_query: Optional[str] = None
        self._current_where: Optional[str] = None
        self._current_params: List[object] = []
        self._offset = 0
        self._results_cache: list[dict[str, object]] = []

        self._thumb_pool = QThreadPool(self)
        self._thumb_pool.setMaxThreadCount(min(4, self._thumb_pool.maxThreadCount()))
        self._thumb_signal = _ThumbnailSignal()
        self._thumb_signal.finished.connect(self._apply_thumbnail)
        self._pending_thumbs: set[int] = set()

    def _set_busy(self, busy: bool) -> None:
        self._search_button.setEnabled(not busy)
        if busy:
            self._load_more_button.setEnabled(False)

    def _on_search_clicked(self) -> None:
        query = self._query_input.text().strip()
        self._set_busy(True)
        try:
            fragment = translate_query(query)
        except ValueError as exc:
            self._status_label.setText(str(exc))
            self._set_busy(False)
            return

        self._current_query = query
        self._current_where = fragment.where
        self._current_params = list(fragment.params)
        self._offset = 0
        self._results_cache.clear()
        self._pending_thumbs.clear()
        self._model.removeRows(0, self._model.rowCount())
        self._fetch_results(reset=True)

    def _on_load_more_clicked(self) -> None:
        if not self._current_where:
            return
        self._set_busy(True)
        self._fetch_results(reset=False)

    def _fetch_results(self, *, reset: bool) -> None:
        if not self._current_where:
            self._status_label.setText("Enter a query to search tags.")
            self._set_busy(False)
            return
        try:
            rows = search_files(
                self._conn,
                self._current_where,
                self._current_params,
                limit=self._PAGE_SIZE,
                offset=self._offset,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._status_label.setText(f"Search failed: {exc}")
            self._set_busy(False)
            return

        if reset and not rows:
            self._status_label.setText("No results.")
        elif rows:
            total = self._offset + len(rows)
            message = f"Showing {total} result(s) for '{self._current_query or '*'}'"
            self._status_label.setText(message)
        self._append_rows(rows)
        self._offset += len(rows)
        self._load_more_button.setEnabled(len(rows) == self._PAGE_SIZE)
        self._set_busy(False)

    def _append_rows(self, rows: Iterable[dict[str, object]]) -> None:
        for record in rows:
            path_obj = Path(str(record.get("path", "")))
            self._results_cache.append(record)
            items = [
                QStandardItem(""),
                QStandardItem(path_obj.name),
                QStandardItem(str(path_obj.parent)),
                QStandardItem(self._format_size(record.get("size"))),
                QStandardItem(self._format_dimensions(record.get("width"), record.get("height"))),
                QStandardItem(self._format_mtime(record.get("mtime"))),
                QStandardItem(self._format_tags(record.get("top_tags", []))),
            ]
            items[0].setData(Qt.AlignCenter, Qt.TextAlignmentRole)
            for item in items:
                item.setEditable(False)
            self._model.appendRow(items)
            row_index = self._model.rowCount() - 1
            self._queue_thumbnail(row_index, path_obj)

    def _queue_thumbnail(self, row: int, path: Path) -> None:
        if not path.exists():
            return
        if row in self._pending_thumbs:
            return
        self._pending_thumbs.add(row)
        task = _ThumbnailTask(row, path, self._THUMB_SIZE, self._THUMB_SIZE, self._thumb_signal)
        self._thumb_pool.start(task)

    def _apply_thumbnail(self, row: int, pixmap: QPixmap) -> None:
        self._pending_thumbs.discard(row)
        if row >= self._model.rowCount():
            return
        item = self._model.item(row, 0)
        if item is None or pixmap.isNull():
            return
        item.setData(pixmap, Qt.DecorationRole)
        item.setData(Qt.AlignCenter, Qt.TextAlignmentRole)
        self._results_view.setRowHeight(row, max(self._THUMB_SIZE + 16, pixmap.height() + 16))

    @staticmethod
    def _format_size(value: object) -> str:
        try:
            size = int(value)
        except (TypeError, ValueError):
            return "-"
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.2f} MiB"
        if size >= 1024:
            return f"{size / 1024:.2f} KiB"
        return f"{size} B"

    @staticmethod
    def _format_dimensions(width: object, height: object) -> str:
        try:
            w = int(width)
            h = int(height)
        except (TypeError, ValueError):
            return "-"
        return f"{w}×{h}"

    @staticmethod
    def _format_mtime(value: object) -> str:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return "-"
        try:
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return "-"

    @staticmethod
    def _format_tags(tags: Iterable[tuple[str, float]]) -> str:
        parts = [f"{name} ({score:.2f})" for name, score in tags]
        return ", ".join(parts)

    def _on_row_double_clicked(self, index: QModelIndex) -> None:
        row = index.row()
        if 0 <= row < len(self._results_cache):
            path = Path(str(self._results_cache[row].get("path", "")))
            self._open_in_explorer(path)

    def _open_in_explorer(self, path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", f"/select,{path}"])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except Exception as exc:  # pragma: no cover - defensive
            self._status_label.setText(f"Failed to open file: {exc}")

    def display_results(self, rows: Iterable[tuple[str, list[object]]]) -> None:
        """Legacy hook retained for compatibility with earlier UI code."""
        self._model.removeRows(0, self._model.rowCount())
        self._results_cache.clear()
        self._pending_thumbs.clear()
        for where_stmt, params in rows:
            stub = [QStandardItem("") for _ in range(self._model.columnCount())]
            if stub:
                stub[1].setText(where_stmt)
            if len(stub) > 2:
                stub[2].setText(str(params))
            for item in stub:
                item.setEditable(False)
            self._model.appendRow(stub)


__all__ = ["TagsTab"]
