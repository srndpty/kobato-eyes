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
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QStackedWidget,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.pipeline import run_index_once
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

    def run(self) -> None:  # noqa: D401
        pixmap = get_thumbnail(self._path, self._width, self._height)
        self._signal.finished.emit(self._row, pixmap)


class TagsTab(QWidget):
    """Provide a search bar and tabular or grid results for tag queries."""

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

        self._placeholder = QWidget(self)
        placeholder_layout = QVBoxLayout(self._placeholder)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label = QLabel("No results yet. Try indexing your library.", self._placeholder)
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_button = QPushButton("Index now", self._placeholder)
        placeholder_layout.addWidget(self._placeholder_label)
        placeholder_layout.addWidget(self._placeholder_button)

        self._table_button = QToolButton(self)
        self._table_button.setText("Table")
        self._table_button.setCheckable(True)
        self._grid_button = QToolButton(self)
        self._grid_button.setText("Grid")
        self._grid_button.setCheckable(True)
        self._table_button.setChecked(True)
        toggle_group = QButtonGroup(self)
        toggle_group.setExclusive(True)
        toggle_group.addButton(self._table_button)
        toggle_group.addButton(self._grid_button)

        self._stack = QStackedWidget(self)

        headers = [
            "Thumb",
            "File name",
            "Folder",
            "Size",
            "Dim",
            "Modified",
            "Top 5 Tags",
        ]
        self._table_model = QStandardItemModel(0, len(headers), self)
        self._table_model.setHorizontalHeaderLabels(headers)
        self._table_view = QTableView(self)
        self._table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_view.doubleClicked.connect(self._on_table_double_clicked)
        self._table_view.setModel(self._table_model)
        self._table_view.horizontalHeader().setStretchLastSection(True)
        self._table_view.setIconSize(QSize(self._THUMB_SIZE, self._THUMB_SIZE))

        self._grid_model = QStandardItemModel(self)
        self._grid_view = QListView(self)
        self._grid_view.setViewMode(QListView.ViewMode.IconMode)
        self._grid_view.setResizeMode(QListView.ResizeMode.Adjust)
        self._grid_view.setMovement(QListView.Movement.Static)
        self._grid_view.setSpacing(16)
        self._grid_view.setWrapping(True)
        self._grid_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._grid_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._grid_view.setIconSize(QSize(self._THUMB_SIZE, self._THUMB_SIZE))
        self._grid_view.setGridSize(QSize(self._THUMB_SIZE + 48, self._THUMB_SIZE + 72))
        self._grid_view.doubleClicked.connect(self._on_grid_double_clicked)
        self._grid_view.setModel(self._grid_model)

        self._stack.addWidget(self._placeholder)
        self._stack.addWidget(self._table_view)
        self._stack.addWidget(self._grid_view)

        search_layout = QHBoxLayout()
        search_layout.addWidget(self._query_input)
        search_layout.addWidget(self._search_button)

        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(self._table_button)
        toggle_layout.addWidget(self._grid_button)
        toggle_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(search_layout)
        layout.addLayout(toggle_layout)
        layout.addWidget(self._status_label)
        layout.addWidget(self._stack)
        layout.addWidget(self._load_more_button)

        self._search_button.clicked.connect(self._on_search_clicked)
        self._load_more_button.clicked.connect(self._on_load_more_clicked)
        self._query_input.returnPressed.connect(self._on_search_clicked)
        self._table_button.toggled.connect(self._on_table_toggled)
        self._grid_button.toggled.connect(self._on_grid_toggled)
        self._placeholder_button.clicked.connect(self._on_index_now)

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

        self._show_placeholder(True)

    def _show_placeholder(self, show: bool) -> None:
        if show:
            self._stack.setCurrentWidget(self._placeholder)
            self._load_more_button.setEnabled(False)
        else:
            target = self._table_view if self._table_button.isChecked() else self._grid_view
            self._stack.setCurrentWidget(target)

    def _on_index_now(self) -> None:
        db_row = self._conn.execute("PRAGMA database_list").fetchone()
        db_file = Path(db_row[2]) if db_row and db_row[2] else Path("kobato-eyes.db")
        run_index_once(db_file)
        self._status_label.setText("Indexing requested.")

    def _on_table_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._table_view)
            self._grid_button.setChecked(False)

    def _on_grid_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._grid_view)
            self._table_button.setChecked(False)

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
        self._table_model.removeRows(0, self._table_model.rowCount())
        self._grid_model.removeRows(0, self._grid_model.rowCount())
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
            self._show_placeholder(True)
            return
        try:
            rows = search_files(
                self._conn,
                self._current_where,
                self._current_params,
                limit=self._PAGE_SIZE,
                offset=self._offset,
            )
        except Exception as exc:  # pragma: no cover
            self._status_label.setText(f"Search failed: {exc}")
            self._set_busy(False)
            return

        if reset and not rows:
            self._status_label.setText("No results. Try indexing your library.")
            self._show_placeholder(True)
        else:
            total = self._offset + len(rows)
            self._status_label.setText(f"Showing {total} result(s) for '{self._current_query or '*'}'")
            self._show_placeholder(False)
        self._append_rows(rows)
        self._offset += len(rows)
        self._load_more_button.setEnabled(len(rows) == self._PAGE_SIZE)
        self._set_busy(False)

    def _append_rows(self, rows: Iterable[dict[str, object]]) -> None:
        for record in rows:
            row_index = len(self._results_cache)
            self._results_cache.append(record)
            path_obj = Path(str(record.get("path", "")))

            table_items = [
                QStandardItem(""),
                QStandardItem(path_obj.name),
                QStandardItem(str(path_obj.parent)),
                QStandardItem(self._format_size(record.get("size"))),
                QStandardItem(self._format_dimensions(record.get("width"), record.get("height"))),
                QStandardItem(self._format_mtime(record.get("mtime"))),
                QStandardItem(self._format_tags(record.get("top_tags", []))),
            ]
            table_items[0].setData(Qt.AlignCenter, Qt.TextAlignmentRole)
            for item in table_items:
                item.setEditable(False)
            self._table_model.appendRow(table_items)
            self._table_view.setRowHeight(row_index, self._THUMB_SIZE + 16)

            grid_item = QStandardItem(self._format_grid_text(path_obj.name, record.get("top_tags", [])))
            grid_item.setEditable(False)
            grid_item.setData(row_index, Qt.UserRole)
            grid_item.setData(Qt.AlignCenter, Qt.TextAlignmentRole)
            grid_item.setSizeHint(QSize(self._THUMB_SIZE + 48, self._THUMB_SIZE + 72))
            self._grid_model.appendRow(grid_item)

            if path_obj.exists():
                self._queue_thumbnail(row_index, path_obj)

    def _queue_thumbnail(self, row: int, path: Path) -> None:
        if row in self._pending_thumbs:
            return
        self._pending_thumbs.add(row)
        task = _ThumbnailTask(row, path, self._THUMB_SIZE, self._THUMB_SIZE, self._thumb_signal)
        self._thumb_pool.start(task)

    def _apply_thumbnail(self, row: int, pixmap: QPixmap) -> None:
        self._pending_thumbs.discard(row)
        if row < self._table_model.rowCount():
            table_item = self._table_model.item(row, 0)
            if table_item is not None:
                table_item.setData(pixmap, Qt.DecorationRole)
                table_item.setData(Qt.AlignCenter, Qt.TextAlignmentRole)
                self._table_view.setRowHeight(row, max(self._THUMB_SIZE + 16, pixmap.height() + 16))
        if row < self._grid_model.rowCount():
            grid_item = self._grid_model.item(row)
            if grid_item is not None:
                grid_item.setData(pixmap, Qt.DecorationRole)

    def _on_table_double_clicked(self, index: QModelIndex) -> None:
        self._open_row(index.row())

    def _on_grid_double_clicked(self, index: QModelIndex) -> None:
        stored_row = index.data(Qt.UserRole)
        row = int(stored_row) if stored_row is not None else index.row()
        self._open_row(row)

    def _open_row(self, row: int) -> None:
        if 0 <= row < len(self._results_cache):
            path = Path(str(self._results_cache[row].get("path", "")))
            self._open_in_explorer(path)

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

    @staticmethod
    def _format_grid_text(name: str, tags: Iterable[tuple[str, float]]) -> str:
        tag_names = [tag for tag, _ in tags][:2]
        subtitle = ", ".join(tag_names)
        return f"{name}\n{subtitle}" if subtitle else name

    def _open_in_explorer(self, path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", f"/select,{path}"])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except Exception as exc:  # pragma: no cover
            self._status_label.setText(f"Failed to open file: {exc}")

    def display_results(self, rows: Iterable[tuple[str, list[object]]]) -> None:
        """Legacy hook retained for backwards compatibility."""
        self._table_model.removeRows(0, self._table_model.rowCount())
        self._grid_model.removeRows(0, self._grid_model.rowCount())
        self._results_cache.clear()
        self._pending_thumbs.clear()
        if rows:
            self._show_placeholder(False)
        else:
            self._show_placeholder(True)
        for where_stmt, params in rows:
            table_stub = [QStandardItem("") for _ in range(self._table_model.columnCount())]
            if len(table_stub) > 1:
                table_stub[1].setText(where_stmt)
            if len(table_stub) > 2:
                table_stub[2].setText(str(params))
            for item in table_stub:
                item.setEditable(False)
            self._table_model.appendRow(table_stub)
            grid_item = QStandardItem(where_stmt)
            grid_item.setEditable(False)
            self._grid_model.appendRow(grid_item)


__all__ = ["TagsTab"]
