"""UI for tag-based search in kobato-eyes."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PyQt6.QtCore import (
    QModelIndex,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QPixmap, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStackedWidget,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.config import load_settings
from core.pipeline import IndexProgress, retag_all, retag_query, run_index_once
from core.settings import PipelineSettings
from core.query import translate_query
from db.connection import get_conn
from db.repository import search_files
from utils.image_io import get_thumbnail
from utils.paths import ensure_dirs, get_db_path
from tagger.wd14_onnx import ONNXRUNTIME_MISSING_MESSAGE

logger = logging.getLogger(__name__)


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


class IndexRunnable(QRunnable):
    """Execute ``run_index_once`` on a worker thread with progress reporting."""

    class IndexSignals(QObject):
        progress = pyqtSignal(int, int, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

    def __init__(
        self,
        db_path: Path,
        *,
        settings: PipelineSettings | None = None,
        pre_run: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self._db_path = Path(db_path)
        self._settings = settings
        self._pre_run = pre_run
        self.signals = self.IndexSignals()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the current indexing run."""

        self._cancel_event.set()

    def _emit_progress(self, progress: IndexProgress) -> None:
        label = progress.phase.name.title()
        if progress.total < 0 and progress.message:
            label = progress.message
        self.signals.progress.emit(progress.done, progress.total, label)

    def run(self) -> None:  # noqa: D401
        try:
            extra: dict[str, object] = {}
            if self._pre_run is not None:
                extra = self._pre_run()
            stats = run_index_once(
                self._db_path,
                settings=self._settings,
                progress_cb=self._emit_progress,
                is_cancelled=self._cancel_event.is_set,
            )
            if extra:
                stats.update(extra)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Indexing failed for database %s", self._db_path)
            self.signals.error.emit(str(exc))
        else:
            self.signals.finished.emit(stats)


class TagsTab(QWidget):
    """Provide a search bar and tabular or grid results for tag queries."""

    _PAGE_SIZE = 200
    _THUMB_SIZE = 128

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._query_input = QLineEdit(self)
        self._query_input.setPlaceholderText("Search tags…")
        self._search_button = QPushButton("Search", self)
        self._retag_menu = QMenu("Retag with current model", self)
        self._retag_all_action = self._retag_menu.addAction("All library")
        self._retag_results_action = self._retag_menu.addAction("Current results")
        self._retag_button = QToolButton(self)
        self._retag_button.setText("Retag…")
        self._retag_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._retag_button.setMenu(self._retag_menu)
        self._load_more_button = QPushButton("Load more", self)
        self._load_more_button.setEnabled(False)
        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)

        self._debug_group = QGroupBox("Debug SQL", self)
        self._debug_group.setCheckable(True)
        self._debug_group.setChecked(False)
        self._debug_group.setVisible(False)
        debug_layout = QVBoxLayout(self._debug_group)
        self._debug_where = QLabel("WHERE: 1=1", self._debug_group)
        self._debug_params = QLabel("Params: []", self._debug_group)
        debug_layout.addWidget(self._debug_where)
        debug_layout.addWidget(self._debug_params)

        self._placeholder = QWidget(self)
        placeholder_layout = QVBoxLayout(self._placeholder)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label = QLabel(
            "No results yet. Try indexing your library.", self._placeholder
        )
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
        search_layout.addWidget(self._retag_button)

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
        layout.addWidget(self._debug_group)

        self._search_button.clicked.connect(self._on_search_clicked)
        self._load_more_button.clicked.connect(self._on_load_more_clicked)
        self._query_input.returnPressed.connect(self._on_search_clicked)
        self._retag_all_action.triggered.connect(self._on_retag_all)
        self._retag_results_action.triggered.connect(self._on_retag_results)
        self._table_button.toggled.connect(self._on_table_toggled)
        self._grid_button.toggled.connect(self._on_grid_toggled)
        self._placeholder_button.clicked.connect(self._on_index_now)

        ensure_dirs()
        self._db_display = str(get_db_path())
        self._conn = get_conn(get_db_path())
        self._db_path = self._resolve_db_path()
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

        self._index_pool = QThreadPool(self)
        self._index_pool.setMaxThreadCount(1)
        self._search_busy = False
        self._indexing_active = False
        self._retag_active = False
        self._can_load_more = False
        self._progress_dialog: QProgressDialog | None = None
        self._current_index_task: IndexRunnable | None = None

        self._toast_label = QLabel("", self)
        self._toast_label.setObjectName("toastLabel")
        self._toast_label.setStyleSheet(
            "#toastLabel {"
            "color: white;"
            "background-color: rgba(0, 0, 0, 180);"
            "border-radius: 6px;"
            "padding: 8px 12px;"
            "}"
        )
        self._toast_label.setVisible(False)
        self._toast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self._toast_label.setVisible(False))

        self._show_placeholder(True)
        self._update_control_states()

    def _show_placeholder(self, show: bool) -> None:
        if show:
            self._stack.setCurrentWidget(self._placeholder)
            self._can_load_more = False
        else:
            target = self._table_view if self._table_button.isChecked() else self._grid_view
            self._stack.setCurrentWidget(target)
        self._update_control_states()

    def _on_index_now(self) -> None:
        if self._indexing_active:
            return
        self._retag_active = False
        self._db_path = self._resolve_db_path()
        task = IndexRunnable(self._db_path)
        self._start_indexing_task(task)

    def _run_retag(
        self,
        *,
        predicate: str | None,
        params: Sequence[object] | None = None,
        force_all: bool = False,
    ) -> None:
        if self._indexing_active:
            return
        self._db_path = self._resolve_db_path()
        settings = load_settings()
        params_list = list(params or [])

        def _pre_run() -> dict[str, object]:
            if predicate is None:
                marked = retag_all(self._db_path, force=force_all, settings=settings)
            else:
                marked = retag_query(self._db_path, predicate, params_list)
            return {"retagged_marked": marked}

        task = IndexRunnable(
            self._db_path,
            settings=settings,
            pre_run=_pre_run,
        )
        self._retag_active = True
        self._start_indexing_task(task)

    def _on_retag_all(self) -> None:
        if self._indexing_active:
            return
        answer = QMessageBox.question(
            self,
            "Retag all files",
            (
                "Retagging the entire library may take a long time.\n"
                "Do you want to continue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._run_retag(predicate=None, params=None, force_all=True)

    def _on_retag_results(self) -> None:
        if self._indexing_active:
            return
        if not self._current_where:
            self._show_toast("Search results are required before retagging.")
            return
        self._run_retag(
            predicate=self._current_where,
            params=list(self._current_params),
            force_all=False,
        )

    def _start_indexing_task(self, task: IndexRunnable) -> None:
        self._current_index_task = task
        task.signals.progress.connect(self._handle_index_progress)
        task.signals.finished.connect(self._handle_index_finished)
        task.signals.error.connect(self._handle_index_failed)
        self._handle_index_started()
        self._progress_dialog = self._create_progress_dialog()
        self._index_pool.start(task)

    def _create_progress_dialog(self) -> QProgressDialog:
        dialog = QProgressDialog("Preparing…", "Cancel", 0, 0, self)
        dialog.setWindowTitle("Retagging" if self._retag_active else "Indexing")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoReset(False)
        dialog.setAutoClose(False)
        dialog.canceled.connect(self._cancel_indexing)
        dialog.show()
        return dialog

    def _cancel_indexing(self) -> None:
        if self._current_index_task is not None:
            self._current_index_task.cancel()
        prefix = "Retagging" if self._retag_active else "Indexing"
        self._status_label.setText(f"{prefix} cancelling…")
        if self._progress_dialog is not None:
            self._progress_dialog.setLabelText("Cancelling…")

    def _handle_index_progress(self, done: int, total: int, label: str) -> None:
        if self._progress_dialog is None:
            return
        if total < 0:
            self._progress_dialog.setRange(0, 0)
            self._progress_dialog.setLabelText(label)
            return
        maximum = max(total, 0)
        value = max(0, min(done, total))
        self._progress_dialog.setRange(0, maximum)
        self._progress_dialog.setValue(value)
        if total > 0:
            percent = min(100, (value * 100) // total)
        else:
            percent = 100 if value else 0
        self._progress_dialog.setLabelText(f"{label}: {value}/{total} ({percent}%)")

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.hide()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        self._current_index_task = None

    def _on_table_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._table_view)
            self._grid_button.setChecked(False)

    def _on_grid_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._grid_view)
            self._table_button.setChecked(False)

    def _set_busy(self, busy: bool) -> None:
        self._search_busy = busy
        if busy:
            self._can_load_more = False
        self._update_control_states()

    def _on_search_clicked(self) -> None:
        query = self._query_input.text().strip()
        self._set_busy(True)
        try:
            fragment = translate_query(query, file_alias="f")
        except ValueError as exc:
            self._status_label.setText(str(exc))
            self._set_busy(False)
            return

        self._debug_where.setText(f"WHERE: {fragment.where}")
        self._debug_params.setText(f"Params: {fragment.params}")
        self._debug_group.setVisible(bool(fragment.where.strip() and fragment.where.strip() != "1=1"))

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
        self._can_load_more = len(rows) == self._PAGE_SIZE
        self._set_busy(False)
        self._update_control_states()

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
            table_items[0].setData(
                Qt.AlignmentFlag.AlignCenter, Qt.ItemDataRole.TextAlignmentRole
            )
            for item in table_items:
                item.setEditable(False)
            self._table_model.appendRow(table_items)
            self._table_view.setRowHeight(row_index, self._THUMB_SIZE + 16)

            grid_item = QStandardItem(self._format_grid_text(path_obj.name, record.get("top_tags", [])))
            grid_item.setEditable(False)
            grid_item.setData(row_index, Qt.ItemDataRole.UserRole)
            grid_item.setData(
                Qt.AlignmentFlag.AlignCenter, Qt.ItemDataRole.TextAlignmentRole
            )
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
                table_item.setData(pixmap, Qt.ItemDataRole.DecorationRole)
                table_item.setData(
                    Qt.AlignmentFlag.AlignCenter, Qt.ItemDataRole.TextAlignmentRole
                )
                self._table_view.setRowHeight(row, max(self._THUMB_SIZE + 16, pixmap.height() + 16))
        if row < self._grid_model.rowCount():
            grid_item = self._grid_model.item(row)
            if grid_item is not None:
                grid_item.setData(pixmap, Qt.ItemDataRole.DecorationRole)

    def _on_table_double_clicked(self, index: QModelIndex) -> None:
        self._open_row(index.row())

    def _on_grid_double_clicked(self, index: QModelIndex) -> None:
        stored_row = index.data(Qt.ItemDataRole.UserRole)
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


    def _update_control_states(self) -> None:
        search_enabled = not self._search_busy and not self._indexing_active
        input_enabled = not self._indexing_active and not self._search_busy
        self._search_button.setEnabled(search_enabled)
        self._query_input.setEnabled(input_enabled)
        self._load_more_button.setEnabled(
            self._can_load_more and not self._indexing_active and not self._search_busy
        )
        self._placeholder_button.setEnabled(not self._indexing_active)
        self._table_button.setEnabled(not self._indexing_active)
        self._grid_button.setEnabled(not self._indexing_active)
        retag_enabled = not self._indexing_active and not self._search_busy
        self._retag_button.setEnabled(retag_enabled)
        self._retag_results_action.setEnabled(bool(self._current_where) and retag_enabled)

    def _handle_index_started(self) -> None:
        self._indexing_active = True
        if self._retag_active:
            self._status_label.setText("Retagging…")
        else:
            self._status_label.setText("Indexing…")
        self._update_control_states()

    def _handle_index_finished(self, stats: dict[str, object]) -> None:
        self._close_progress_dialog()
        self._indexing_active = False
        elapsed = float(stats.get("elapsed_sec", 0.0) or 0.0)
        cancelled = bool(stats.get("cancelled", False))
        prefix = "Retagging" if self._retag_active else "Indexing"
        if cancelled:
            self._status_label.setText(f"{prefix} cancelled after {elapsed:.2f}s.")
            self._show_toast(f"{prefix} cancelled.")
            self._retag_active = False
            self._update_control_states()
            return
        if self._retag_active:
            self._status_label.setText(f"Retagging complete in {elapsed:.2f}s.")
        else:
            self._status_label.setText(f"Indexing complete in {elapsed:.2f}s.")
        tagger_name = str(stats.get("tagger_name") or "unknown")
        message = (
            f"Indexed: {int(stats.get('scanned', 0))} files / "
            f"Tagged: {int(stats.get('tagged', 0))} / "
            f"Embedded: {int(stats.get('embedded', 0))}"
        )
        retagged = int(stats.get("retagged", 0) or 0)
        requested = int(stats.get("retagged_marked", retagged) or 0)
        if self._retag_active:
            if requested and requested != retagged:
                message += f" / Retagged: {retagged}/{requested}"
            else:
                message += f" / Retagged: {retagged}"
        elif retagged:
            message += f" / Retagged: {retagged}"
        message += f" (tagger: {tagger_name})"
        self._show_toast(message)
        self._retag_active = False
        self._update_control_states()
        QTimer.singleShot(0, self._on_search_clicked)

    def _handle_index_failed(self, message: str) -> None:
        self._close_progress_dialog()
        self._indexing_active = False
        if message == ONNXRUNTIME_MISSING_MESSAGE:
            error_text = message
        else:
            prefix = "Retagging" if self._retag_active else "Indexing"
            error_text = f"{prefix} failed (DB: {self._db_display}): {message}"
        self._status_label.setText(error_text)
        self._show_toast(error_text)
        self._retag_active = False
        self._update_control_states()

    def _resolve_db_path(self) -> Path:
        db_row = self._conn.execute("PRAGMA database_list").fetchone()
        literal = db_row[2] if db_row else None
        if literal and literal not in {":memory:", ""}:
            path = Path(literal).expanduser()
            self._db_display = str(path)
            return path
        if literal == ":memory:":
            self._db_display = ":memory:"
            return Path(get_db_path()).expanduser()
        fallback = Path(get_db_path()).expanduser()
        self._db_display = str(fallback)
        return fallback

    def _show_toast(self, message: str, *, timeout_ms: int = 4000) -> None:
        self._toast_timer.stop()
        self._toast_label.setText(message)
        self._toast_label.adjustSize()
        width = self.width()
        height = self.height()
        label_width = self._toast_label.width()
        label_height = self._toast_label.height()
        x = max(0, (width - label_width) // 2)
        y = max(0, height - label_height - 16)
        self._toast_label.move(x, y)
        self._toast_label.setVisible(True)
        self._toast_timer.start(timeout_ms)

__all__ = ["TagsTab"]
