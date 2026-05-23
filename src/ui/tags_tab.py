"""UI for tag-based search in kobato-eyes."""

from __future__ import annotations

import itertools
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from contextlib import AbstractContextManager, closing, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PyQt6.QtCore import (
    QAbstractListModel,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QDesktopServices, QKeyEvent, QKeySequence, QPixmap, QShortcut, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCompleter,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.config import PipelineSettings
from core.pipeline import IndexProgress, run_index_once
from db.connection import get_conn
from tagger import labels_util
from tagger.base import TagCategory
from tagger.wd14_onnx import ONNXRUNTIME_MISSING_MESSAGE
from ui.autocomplete import (
    abbreviate_count,
    completion_search_prefix,
    extract_completion_token,
    replace_completion_token,
)
from ui.file_actions import trash_path
from ui.index_lifecycle import (
    connection_retry_action,
    index_cancel_status,
    index_started_status,
    plan_index_failed,
    plan_index_finished,
)
from ui.index_tasks import IndexRunnable
from ui.result_delegates import GridThumbDelegate
from ui.result_delegates import HighlightDelegate as _HighlightDelegate
from ui.result_delegates import WrappingItemDelegate as _WrappingItemDelegate
from ui.search_worker import SearchWorker
from ui.tag_rendering import _SCORE_COLOR, _TAG_LIST_ROLE, TagDisplayEntry
from ui.tag_rendering import coerce_category as _coerce_category
from ui.tag_rendering import filter_tags_by_threshold as _filter_tags_by_threshold
from ui.tag_stats import TagStatsDialog
from ui.tags_control_state import TagsActivityState, compute_tags_control_availability
from ui.thumbnail_tasks import ThumbnailSignal as _ThumbnailSignal
from ui.thumbnail_tasks import ThumbnailTask as _ThumbnailTask
from ui.viewmodels import TagsSearchState, TagsViewModel
from ui.widgets.spinner_overlay import SpinnerOverlay

logger = logging.getLogger(__name__)


_CATEGORY_PREFIXES = [f"{category.name.lower()}:" for category in TagCategory]
_RESERVED_COMPLETIONS = ["category:", *_CATEGORY_PREFIXES]
_RESERVED = {"and", "or", "not"}
_PREFIXES = (
    "category:",
    "general:",
    "character:",
    "copyright:",
    "artist:",
    "meta:",
    "rating:",
)


class _ElidingLabel(QLabel):
    """幅に収まらないテキストを…で中間省略し、フルテキストはツールチップに出す。"""

    def __init__(self, text: str = "", *, mode=Qt.TextElideMode.ElideMiddle, parent: QWidget | None = None):
        super().__init__(text, parent)
        self._full = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(480)  # ダイアログ幅の目安（お好みで）
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def set_full_text(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._apply_elide()

    def resizeEvent(self, ev):  # type: ignore[override]
        super().resizeEvent(ev)
        self._apply_elide()

    def _apply_elide(self) -> None:
        w = max(50, self.width())
        fm = self.fontMetrics()
        elided = fm.elidedText(self._full, self._mode, w)
        super().setText(elided)


# --- ワーカーシグナル ---------------------------------------------------
class _CopySignals(QObject):
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(str, int, int)  # dest_dir, ok_count, ng_count
    error = pyqtSignal(str)


class _DeleteResultSignals(QObject):
    """Signals emitted by a background search-result deletion task."""

    finished = pyqtSignal(list, list)


class _DeleteResultRunnable(QRunnable):
    """Move result files to trash and mark their DB rows absent."""

    def __init__(
        self,
        view_model: TagsViewModel,
        db_path: Path,
        *,
        entries: Sequence[tuple[int, Path]],
    ) -> None:
        super().__init__()
        self._view_model = view_model
        self._db_path = Path(db_path)
        self._entries = [(int(file_id), Path(path)) for file_id, path in entries]
        self.signals = _DeleteResultSignals()

    def run(self) -> None:
        """Execute the delete workflow off the GUI thread."""

        successes: list[tuple[int, str]] = []
        failures: list[tuple[int, str, str]] = []
        for file_id, path in self._entries:
            try:
                trash_path(path)
                successes.append((file_id, str(path)))
            except Exception as exc:
                failures.append((file_id, str(path), str(exc)))

        if successes:
            success_ids = [file_id for file_id, _ in successes]
            success_paths = {file_id: path for file_id, path in successes}
            db_successes: list[tuple[int, str]] = []
            db_failures: list[tuple[int, str, str]] = []
            try:
                with closing(self._view_model.open_connection(self._db_path)) as conn:
                    self._view_model.mark_files_absent(conn, success_ids)
                db_successes = successes
            except Exception as exc:
                message = str(exc)
                db_failures = [(file_id, success_paths[file_id], message) for file_id in success_ids]
            successes = db_successes
            failures.extend(db_failures)

        self.signals.finished.emit(successes, failures)


def _format_delete_confirmation(paths: Sequence[Path]) -> str:
    """Return confirmation text for deleting selected search results."""

    if len(paths) == 1:
        return f"Move this image to the trash and remove it from search results?\n\n{paths[0]}"
    preview = "\n".join(str(path) for path in paths[:5])
    suffix = "" if len(paths) <= 5 else f"\n... and {len(paths) - 5} more"
    return f"Move {len(paths)} images to the trash and remove them from search results?\n\n{preview}{suffix}"


def _format_deleting_status(paths: Sequence[Path]) -> str:
    """Return status text for an active delete operation."""

    if len(paths) == 1:
        return f"Deleting {paths[0].name}…"
    return f"Deleting {len(paths)} images…"


def _format_delete_success(successes: Sequence[tuple[int, str]], total: int, remaining: int, query_label: str) -> str:
    """Return status text after a delete operation finishes."""

    if len(successes) == 1 and total == 1:
        try:
            name = Path(successes[0][1]).name
        except IndexError:
            name = "image"
        return f"Deleted {name}. Showing {remaining} result(s) for '{query_label}'"
    return f"Deleted {len(successes)}/{total} image(s). Showing {remaining} result(s) for '{query_label}'"


# --- バックグラウンドコピー ---------------------------------------------
class _CopyRunnable(QRunnable):
    def __init__(
        self,
        view_model: TagsViewModel,
        db_path: Path,
        query: str,
        dest_dir: Path,
    ) -> None:
        super().__init__()
        self._view_model = view_model
        self.db_path = db_path
        self.query = query
        self.dest_dir = dest_dir
        self.signals = _CopySignals()

    def _unique_dest(self, name: str) -> Path:
        """同名ファイルがある場合は _2, _3... と連番で回避。"""
        dest = self.dest_dir / name
        if not dest.exists():
            return dest
        stem = dest.stem
        suf = dest.suffix
        for i in itertools.count(2):
            cand = self.dest_dir / f"{stem}_{i}{suf}"
            if not cand.exists():
                return cand

    def run(self) -> None:
        try:
            # 1) パス列挙
            with self._view_model.open_connection(self.db_path) as conn:
                paths = self._view_model.iter_paths_for_search(conn, self.query)
            total = len(paths)
            self.signals.progress.emit(0, total)
            ok = ng = 0

            # 2) コピー
            for idx, p in enumerate(paths, start=1):
                try:
                    src = Path(p)
                    if src.exists():
                        dst = self._unique_dest(src.name)
                        shutil.copy2(src, dst)
                        ok += 1
                    else:
                        ng += 1
                except Exception:
                    ng += 1
                self.signals.progress.emit(idx, total)

            self.signals.finished.emit(str(self.dest_dir), ok, ng)
        except Exception as e:
            self.signals.error.emit(str(e))


class TagsTab(QWidget):
    """Provide a search bar and tabular or grid results for tag queries."""

    class TagListModel(QAbstractListModel):
        """Simple list model backed by a list of tag metadata."""

        NAME_ROLE = Qt.ItemDataRole.UserRole + 1
        COUNT_ROLE = Qt.ItemDataRole.UserRole + 2

        def __init__(
            self,
            items: Sequence[labels_util.TagMeta] | None = None,
            parent: QObject | None = None,
        ) -> None:
            super().__init__(parent)
            self._items = list(items or [])
            self._display_prefix = ""

        def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
            if parent and parent.isValid():
                return 0
            return len(self._items)

        def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
            if not index.isValid():
                return None
            try:
                item = self._items[index.row()]
            except IndexError:
                return None
            if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
                count_text = abbreviate_count(item.count)
                name = f"{self._display_prefix}{item.name}"
                return f"{name} ({count_text})" if count_text else name
            if role == int(self.NAME_ROLE):
                return item.name
            if role == int(self.COUNT_ROLE):
                try:
                    return int(item.count or 0)
                except (TypeError, ValueError):
                    return 0
            return None

        def roleNames(self) -> dict[int, bytes]:  # type: ignore[override]
            roles = dict(super().roleNames())
            roles[int(self.NAME_ROLE)] = b"name"
            roles[int(self.COUNT_ROLE)] = b"count"
            return roles

        def reset_with(self, items: Sequence[labels_util.TagMeta], *, display_prefix: str = "") -> None:
            self.beginResetModel()
            self._items = list(items)
            self._display_prefix = display_prefix
            self.endResetModel()

    _PAGE_SIZE = 200
    _THUMB_SIZE = 128

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        view_model: TagsViewModel | None = None,
        search_chunk_size: int | None = None,
        search_chunk_delay: float = 0.0,
    ) -> None:
        super().__init__(parent)
        self._view_model = view_model or TagsViewModel(
            self,
            connection_factory=lambda path: get_conn(path),
            run_index_once=lambda *args, **kwargs: run_index_once(*args, **kwargs),
        )
        self._search_chunk_size = max(1, int(search_chunk_size or self._PAGE_SIZE))
        self._search_chunk_delay = max(0.0, float(search_chunk_delay))
        self._query_edit = QLineEdit(self)
        self._query_edit.setPlaceholderText("Search tags…")
        self._tag_model = self.TagListModel(parent=self)
        self._completer = QCompleter(self._tag_model, self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setCompletionRole(int(self._tag_model.NAME_ROLE))
        self._completer.activated[QModelIndex].connect(self._on_completion_activated)
        # self._query_edit.setCompleter(self._completer)
        self._completer.setWidget(self._query_edit)  # 自前でやる場合はこちらに置き換え

        # completer 作成直後あたり
        self._query_edit.setObjectName("queryEdit")
        self._completer.setWidget(self._query_edit)  # ← これ大事
        self._query_edit.setCompleter(None)  # ← QLineEdit のデフォ補完は無効化

        # event filter を “全部” に入れる（行・popup・viewport・アプリ全体）
        self._query_edit.installEventFilter(self)
        self._completer.popup().installEventFilter(self)
        self._completer.popup().viewport().installEventFilter(self)

        QApplication.instance().installEventFilter(self)

        self._suppress_return_once = False  # 直後の Enter を1回だけ無効化したい時用

        # Tab で補完確定（未選択なら第1候補）
        self._shortcut_tab = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        self._shortcut_tab.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_tab.activated.connect(
            lambda: (self._accept_completion(default_if_none=True) if self._completer.popup().isVisible() else None)
        )

        # Enter / Return は「候補が出ていれば補完、出ていなければ検索」に振り分け
        self._shortcut_return = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        self._shortcut_return.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_return.activated.connect(self._on_return_shortcut)

        self._shortcut_enter = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        self._shortcut_enter.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_enter.activated.connect(self._on_return_shortcut)

        # =============================================
        self._autocomplete_timer = QTimer(self)
        self._autocomplete_timer.setSingleShot(True)
        self._autocomplete_timer.setInterval(150)
        self._autocomplete_timer.timeout.connect(self._refresh_completions)
        self._query_edit.textEdited.connect(self._on_query_text_edited)
        self._all_tags: list[labels_util.TagMeta] = []
        self._completion_candidates: list[labels_util.TagMeta] = []
        self._pending_completion_text = ""
        self._current_completion_range: tuple[int, int] = (0, 0)
        self._update_completion_candidates()
        self._search_button = QPushButton("Search", self)
        self._retag_menu = QMenu("Retag with current model", self)
        self._retag_all_action = self._retag_menu.addAction("All library")
        self._retag_results_action = self._retag_menu.addAction("Current results")
        self._retag_button = QToolButton(self)
        self._retag_button.setText("Retag…")
        self._retag_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._retag_button.setMenu(self._retag_menu)
        self._refresh_button = QPushButton("🔄 Refresh", self)
        self._refresh_button.setToolTip("Scan & tag untagged in this folder (Shift+Click = hard delete missing)")
        # 検索バーのボタン群を並べているあたり（_refresh_button のすぐ右あたりが見栄え良い）
        self._copy_button = QPushButton("Copy results…", self)
        self._copy_button.setEnabled(False)  # 初期は無効
        self._copy_button.clicked.connect(self._on_copy_results_clicked)
        self._open_db_button = QPushButton("Open DB folder", self)
        self._open_db_button.setToolTip("Open the folder containing the database file")
        self._open_db_button.clicked.connect(self._open_db_folder)

        self._load_more_button = QPushButton("Load more", self)
        self._load_more_button.setEnabled(False)
        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)

        self._debug_group = QGroupBox("Debug SQL", self)
        self._debug_group.setCheckable(True)
        self._debug_group.setChecked(False)  # 既定は折りたたみ
        self._debug_group.setVisible(False)  # クエリがない間は非表示のまま

        debug_layout = QVBoxLayout(self._debug_group)
        debug_layout.setContentsMargins(8, 4, 8, 8)

        # ← 中身を直に group に入れず “コンテナ”に入れる
        self._debug_container = QWidget(self._debug_group)
        self._debug_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        inner = QVBoxLayout(self._debug_container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(4)

        self._debug_where = QLabel("WHERE: 1=1", self._debug_container)
        self._debug_where.setWordWrap(True)
        self._debug_params = QLabel("Params: []", self._debug_container)
        self._debug_params.setWordWrap(True)
        # （任意）等幅にしたい場合
        # mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        # self._debug_where.setFont(mono); self._debug_params.setFont(mono)

        inner.addWidget(self._debug_where)
        inner.addWidget(self._debug_params)

        debug_layout.addWidget(self._debug_container)

        # 折りたたみ切り替えハンドラ
        self._debug_group.toggled.connect(self._on_debug_toggled)

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
            "Tags",
        ]
        self._table_model = QStandardItemModel(0, len(headers), self)
        self._table_model.setHorizontalHeaderLabels(headers)
        self._table_view = QTableView(self)
        self._table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_view.doubleClicked.connect(self._on_table_double_clicked)
        # self._table_view.activated.connect(self._on_table_double_clicked)
        self._table_view.setModel(self._table_model)
        self._table_view.horizontalHeader().setStretchLastSection(True)
        self._table_view.setIconSize(QSize(self._THUMB_SIZE, self._THUMB_SIZE))
        self._table_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._table_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._table_view.setWordWrap(True)
        scroll_amount = 36
        self._table_view.verticalScrollBar().setSingleStep(scroll_amount)
        self._table_view.horizontalScrollBar().setSingleStep(scroll_amount)
        self._table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table_view.customContextMenuRequested.connect(self._on_table_context_menu)
        self._table_view.setStyleSheet(f"QTableView, QTableView::item {{color: {_SCORE_COLOR};}}")
        self._delete_table_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self._table_view)
        self._delete_table_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_table_shortcut.activated.connect(self._on_delete_selected_result)

        self._grid_model = QStandardItemModel(self)
        self._grid_view = QListView(self)
        self._grid_view.setViewMode(QListView.ViewMode.IconMode)
        self._grid_view.setResizeMode(QListView.ResizeMode.Adjust)
        self._grid_view.setMovement(QListView.Movement.Static)
        self._grid_view.setSpacing(16)
        self._grid_view.setWrapping(True)
        self._grid_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._grid_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._grid_view.setIconSize(QSize(self._THUMB_SIZE, self._THUMB_SIZE))
        self._grid_view.setGridSize(QSize(self._THUMB_SIZE + 48, self._THUMB_SIZE + 72))
        self._grid_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._grid_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._grid_view.verticalScrollBar().setSingleStep(scroll_amount)
        self._grid_view.horizontalScrollBar().setSingleStep(scroll_amount)
        self._grid_view.doubleClicked.connect(self._on_grid_double_clicked)
        self._grid_view.activated.connect(self._on_grid_double_clicked)
        self._grid_view.setModel(self._grid_model)
        self._grid_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid_view.customContextMenuRequested.connect(self._on_grid_context_menu)
        self._delete_grid_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self._grid_view)
        self._delete_grid_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_grid_shortcut.activated.connect(self._on_delete_selected_result)

        self._highlight_terms: list[str] = []
        self._positive_terms: list[str] = []
        self._relevance_thresholds: dict[int, float] = {}
        self._use_relevance = False
        self._filename_delegate = _WrappingItemDelegate(self._table_view)
        self._folder_delegate = _WrappingItemDelegate(self._table_view)
        self._tags_delegate = _HighlightDelegate(lambda: self._highlight_terms, self._table_view)
        tags_col = self._table_model.columnCount() - 1
        if tags_col >= 0:
            if self._table_model.columnCount() > 2:
                self._table_view.setItemDelegateForColumn(1, self._filename_delegate)
                self._table_view.setItemDelegateForColumn(2, self._folder_delegate)
            self._table_view.setItemDelegateForColumn(tags_col, self._tags_delegate)
        self._grid_delegate = GridThumbDelegate(self._THUMB_SIZE, self._grid_view)
        self._grid_view.setItemDelegate(self._grid_delegate)

        self._stack.addWidget(self._placeholder)
        self._stack.addWidget(self._table_view)
        self._stack.addWidget(self._grid_view)
        self._search_overlay = SpinnerOverlay(self._stack)
        self._search_overlay.hide()

        search_layout = QHBoxLayout()
        search_layout.addWidget(self._query_edit)
        search_layout.addWidget(self._search_button)
        search_layout.addWidget(self._retag_button)
        search_layout.addWidget(self._refresh_button)

        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(self._table_button)
        toggle_layout.addWidget(self._grid_button)
        self._stats_button = QPushButton("Stats", self)
        self._stats_button.setToolTip("Show tag statistics")
        toggle_layout.addWidget(self._stats_button)
        toggle_layout.addWidget(self._copy_button)
        toggle_layout.addWidget(self._open_db_button)
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
        self._query_edit.returnPressed.connect(self._on_search_clicked)
        self._cancel_search_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._cancel_search_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._cancel_search_shortcut.activated.connect(self._cancel_active_search)
        self._retag_all_action.triggered.connect(self._on_retag_all)
        self._retag_results_action.triggered.connect(self._on_retag_results)
        self._table_button.toggled.connect(self._on_table_toggled)
        self._grid_button.toggled.connect(self._on_grid_toggled)
        self._stats_button.clicked.connect(self._open_stats)
        self._placeholder_button.clicked.connect(self._on_index_now)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)

        self._search_state = TagsSearchState()
        self._current_query: Optional[str] = None
        self._current_where: Optional[str] = None
        self._current_params: List[object] = []
        self._results_cache: list[dict[str, object]] = []
        self._tag_thresholds: dict[TagCategory, float] = {}
        self._search_thread: QThread | None = None
        self._search_worker: SearchWorker | None = None

        self._view_model.ensure_directories()
        self._db_display = str(self._view_model.db_path)
        self._conn: sqlite3.Connection | None = None
        self._open_connection()
        self._db_path = self._resolve_db_path()
        self.destroyed.connect(self._cancel_active_search)
        self.destroyed.connect(self._close_connection)
        self._update_thresholds(self._view_model.load_settings())

        self._thumb_pool = QThreadPool(self)
        self._thumb_pool.setMaxThreadCount(min(4, self._thumb_pool.maxThreadCount()))
        self._thumb_signal = _ThumbnailSignal()
        self._thumb_signal.finished.connect(self._apply_thumbnail)
        self._pending_thumbs: set[int] = set()

        self._index_pool = QThreadPool(self)
        self._index_pool.setMaxThreadCount(1)
        self._indexing_active = False
        self._retag_active = False
        self._refresh_active = False
        self._delete_active = False
        self._progress_dialog: QProgressDialog | None = None
        self._current_index_task: IndexRunnable | None = None
        self._active_refresh_folder: Sequence[Path] | None = None
        self._quiesce_guard: AbstractContextManager[None] = nullcontext()
        self._delete_pool = QThreadPool(self)
        self._delete_pool.setMaxThreadCount(1)

        self._toast_label = QLabel("", self)
        self._toast_label.setObjectName("toastLabel")
        self._toast_label.setStyleSheet(
            "#toastLabel {color: white;background-color: rgba(0, 0, 0, 180);border-radius: 6px;padding: 8px 12px;}"
        )
        self._toast_label.setVisible(False)
        self._toast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self._toast_label.setVisible(False))

        self._query_edit.installEventFilter(self)
        self._suppress_return_once = False  # Enter誤発火抑止フラグ
        self._progress_label: _ElidingLabel | None = None

        self._on_debug_toggled(False)
        self._show_placeholder(True)
        self._update_control_states()
        QTimer.singleShot(0, self._initialise_autocomplete)
        QTimer.singleShot(0, self._bootstrap_results_if_any)

    def _open_db_folder(self) -> None:
        try:
            path = getattr(self, "_db_path", None)
            db_path = Path(path) if path else self._resolve_db_path()
            target_dir = db_path if db_path.is_dir() else db_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target_dir))):
                raise RuntimeError("Failed to open directory")
        except Exception as exc:
            QMessageBox.critical(self, "Open DB folder", f"Failed to open folder:\n{exc}")

    def _on_copy_results_clicked(self) -> None:
        # 現在の検索が確定していることを確認
        if not self._current_query or not self._current_where:
            QMessageBox.information(self, "Copy results", "Run a search first.")
            return

        query = self._current_query.strip() or "*"

        # 件数の事前確認（UI表示用）
        try:
            with self._view_model.open_connection(self._db_path) as conn:
                total = len(self._view_model.iter_paths_for_search(conn, query))
        except Exception as e:
            QMessageBox.critical(self, "Copy results", f"Failed to enumerate results:\n{e}")
            return

        if total <= 0:
            QMessageBox.information(self, "Copy results", "No results to copy.")
            return

        dest = self._view_model.make_export_dir(query)
        choice = QMessageBox.question(
            self,
            "Copy results",
            f"Copy {total} file(s) to:\n{dest}\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return

        runnable = _CopyRunnable(self._view_model, self._db_path, query, dest)
        runnable.signals.progress.connect(lambda cur, tot: self._status_label.setText(f"Copying… {cur}/{tot}"))
        runnable.signals.error.connect(lambda msg: QMessageBox.critical(self, "Copy results", msg))

        def _done(dest_dir: str, ok: int, ng: int) -> None:
            self._status_label.setText(f"Copied {ok} file(s). Failed {ng}. → {dest_dir}")
            open_ok = QMessageBox.question(
                self,
                "Copy results",
                f"Done.\nCopied {ok}, Failed {ng}.\nOpen folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if open_ok == QMessageBox.StandardButton.Yes:
                try:
                    if sys.platform.startswith("win"):
                        subprocess.Popen(["explorer", dest_dir])
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", dest_dir])
                    else:
                        subprocess.Popen(["xdg-open", dest_dir])
                except Exception:
                    pass
            self._copy_button.setEnabled(True)

        runnable.signals.finished.connect(_done)
        self._copy_button.setEnabled(False)
        (getattr(self, "_threadpool", None) or QThreadPool.globalInstance()).start(runnable)

    def _on_debug_toggled(self, checked: bool) -> None:
        # 中身の表示・非表示
        self._debug_container.setVisible(checked)

        # 折りたたみ時はヘッダーぶん程度の高さに制限
        if checked:
            self._debug_group.setMaximumHeight(16777215)  # 制限解除
        else:
            header_h = self._debug_group.fontMetrics().height() + 12  # だいたいのヘッダー高さ
            self._debug_group.setMaximumHeight(header_h)

        # レイアウト再計算
        self._debug_group.updateGeometry()
        self.layout().activate()  # ルートレイアウトを再評価

    def _on_return_shortcut(self) -> None:
        if self._completer.popup().isVisible():
            self._accept_completion(default_if_none=True)
            self._suppress_return_once = True
            return
        # 候補が無いときは通常の検索へ（既存の returnPressed を使っているなら何もしない）
        self._on_search_clicked()

    def eventFilter(self, obj, event):
        query_edit = getattr(self, "_query_edit", None)
        if query_edit is not None and obj is query_edit and event.type() == QEvent.Type.KeyPress:
            e: QKeyEvent = event  # type: ignore[assignment]

            key = e.key()
            popup = self._completer.popup()
            popup_visible = bool(popup and popup.isVisible())

            # ↓↑ は completer に任せる（表示している時）。非表示なら通常動作
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                return False

            # Tab: 候補が見えている時は補完確定（未選択なら第1候補）
            if key == Qt.Key.Key_Tab and popup_visible:
                self._accept_completion(default_if_none=True)
                e.accept()
                return True

            # Enter/Return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if popup_visible:
                    # 補完だけ確定して検索は抑止
                    self._accept_completion(default_if_none=True)
                    self._suppress_return_once = True
                    e.accept()
                    return True
                # 直前に補完確定した直後の Enter は無視（検索を抑止）
                if self._suppress_return_once:
                    self._suppress_return_once = False
                    e.accept()
                    return True
                # 候補が無い=確定済みの文字列 → 通常の検索へ
                return False
        return super().eventFilter(obj, event)

    def _accept_completion(self, default_if_none: bool = False) -> None:
        popup = self._completer.popup()
        index = popup.currentIndex() if popup is not None else None
        if (index is None or not index.isValid()) and not default_if_none:
            return
        if (index is None or not index.isValid()) and default_if_none:
            # 第1候補を採用
            index = self._completer.completionModel().index(0, 0)
            if not index.isValid():
                return

        # モデルから「表示名（件数なし）」を取得
        completion = index.data(int(self._tag_model.NAME_ROLE)) or index.data(Qt.ItemDataRole.DisplayRole)
        if not completion:
            return
        text_clean = str(completion)
        text_clean = re.sub(r"\s* \([^)]*\)\s*$", "", text_clean)  # 念のため

        base_text = self._pending_completion_text or self._query_edit.text()
        start, end = self._current_completion_range
        if start > len(base_text) or end > len(base_text):
            token, start, end = extract_completion_token(base_text, len(base_text))

        new_text, cursor = replace_completion_token(base_text, start, end, text_clean)

        def apply():
            block = self._query_edit.blockSignals(True)
            try:
                self._query_edit.setText(new_text)
            finally:
                self._query_edit.blockSignals(block)
            self._query_edit.setCursorPosition(cursor)
            self._pending_completion_text = new_text
            self._hide_completion_popup()
            self._query_edit.setFocus()

        # QLineEditの内部処理が終わった直後に上書き
        QTimer.singleShot(0, apply)

    def _initialise_autocomplete(self) -> None:
        settings = self._view_model.load_settings()
        self.reload_autocomplete(settings)

    def _open_connection(self) -> None:
        if self._conn is not None:
            return
        self._conn = self._view_model.open_connection()

    def _db_has_files(self) -> bool:
        if self._conn is None:
            return False
        try:
            row = self._conn.execute("SELECT 1 FROM files LIMIT 1").fetchone()
        except Exception:
            return False
        return bool(row)

    def _bootstrap_results_if_any(self) -> None:
        if self._db_has_files():
            self._debug_where.setText("WHERE: 1=1\nORDER: f.mtime DESC")
            self._debug_params.setText("Params: []\nRelevance terms: []")
            self._debug_group.setVisible(False)
            self._show_placeholder(False)
            self._current_query = "*"
            self._current_where = "1=1"
            self._current_params = []
            self._highlight_terms = []
            self._positive_terms = []
            self._use_relevance = False
            self._relevance_thresholds = {}
            self._search_state.begin_query()
            self._status_label.setText("Searching…")
            self._search_overlay.show("Loading latest… (Esc to cancel)")
            self._set_busy(True)
            self._start_async_search(reset=True)
        else:
            self._show_placeholder(True)
            self._status_label.setText("No results yet. Click 'Index now' to scan your library.")

    def _close_connection(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        finally:
            self._conn = None

    def prepare_for_database_reset(self) -> None:
        self._cancel_active_search()
        self._close_connection()

    def handle_database_reset(self) -> None:
        self._open_connection()
        self._db_path = self._resolve_db_path()
        self._results_cache.clear()
        self._table_model.removeRows(0, self._table_model.rowCount())
        self._grid_model.removeRows(0, self._grid_model.rowCount())
        self._offset = 0
        self._current_query = ""
        self._current_where = ""
        self._current_params = []
        self._pending_thumbs.clear()
        self._status_label.setText("Database reset. Run indexing to populate results.")
        self._show_placeholder(True)
        self.reload_autocomplete(self._view_model.load_settings())

    def restore_connection(self) -> None:
        self._open_connection()
        self._db_path = self._resolve_db_path()
        self._update_control_states()

    def is_indexing_active(self) -> bool:
        return bool(self._indexing_active)

    def start_indexing_now(self) -> None:
        self._on_index_now()

    def _on_query_text_edited(self, text: str) -> None:
        self._pending_completion_text = text
        if not self._completion_candidates:
            self._tag_model.reset_with([])
            self._hide_completion_popup()
            return
        self._autocomplete_timer.start()
        # 自動検索は行わない（補完リフレッシュのみ）

    def _refresh_completions(self) -> None:
        if not self._completion_candidates:
            self._tag_model.reset_with([])
            self._hide_completion_popup()
            return
        text = self._query_edit.text()
        cursor_position = self._query_edit.cursorPosition()
        token, start, end = extract_completion_token(text, cursor_position)
        self._current_completion_range = (start, end)
        prefix = completion_search_prefix(token).lower()
        if not prefix:
            self._tag_model.reset_with([])
            self._hide_completion_popup()
            return
        matches: list[labels_util.TagMeta] = []
        lower_prefix = prefix
        for candidate in self._completion_candidates:
            if candidate.name.lower().startswith(lower_prefix):
                matches.append(candidate)
        if matches:
            ranked = labels_util.sort_by_popularity(matches)
            limited = ranked[:50]
            display_prefix = "-" if token.startswith("-") else ""
            self._tag_model.reset_with(limited, display_prefix=display_prefix)
            # ★ ここが肝心：QCompleter にも “このトークン” を prefix として教える
            self._completer.setCompletionPrefix(completion_search_prefix(token))
            self._completer.complete()
        else:
            self._tag_model.reset_with([])
            self._hide_completion_popup()

    def _hide_completion_popup(self) -> None:
        popup = self._completer.popup()
        if popup is not None and popup.isVisible():
            popup.hide()

    def _on_completion_activated(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        completion = index.data(int(self._tag_model.NAME_ROLE))
        if not completion:
            completion = index.data(Qt.ItemDataRole.DisplayRole)
            if isinstance(completion, str):
                completion = re.sub(r"\s*\([^)]*\)\s*$", "", completion)
        if not completion:
            return
        completion_text = str(completion)
        # ★ ここがポイント：QCompleter が行全体を置換する前のテキストを使う
        base_text = self._pending_completion_text or self._query_edit.text()
        start, end = self._current_completion_range
        # インデックスがベース文字列からはみ出す場合の保険
        if start > len(base_text) or end > len(base_text):
            token, start, end = replace_completion_token.extract_completion_token(base_text, len(base_text))

        new_text, cursor = replace_completion_token(base_text, start, end, completion_text)
        logger.debug(f"base_text:{base_text}, new_text:{new_text}, cursor:{cursor}")
        block = self._query_edit.blockSignals(True)
        self._query_edit.setText(new_text)
        self._query_edit.blockSignals(block)
        self._query_edit.setCursorPosition(cursor)
        self._pending_completion_text = new_text
        self._autocomplete_timer.stop()
        self._hide_completion_popup()

    def _update_completion_candidates(self) -> None:
        seen: dict[str, labels_util.TagMeta] = {}
        for value in _RESERVED_COMPLETIONS:
            name = value.strip()
            if not name:
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = labels_util.TagMeta(name=name, category=0, count=0)
        for tag in self._all_tags:
            name = tag.name.strip()
            if not name:
                continue
            key = name.lower()
            if key not in seen or int(tag.count or 0) > int(seen[key].count or 0):
                seen[key] = tag
        self._completion_candidates = list(seen.values())
        if not self._completion_candidates:
            self._tag_model.reset_with([])
            self._hide_completion_popup()

    def reload_autocomplete(self, settings: PipelineSettings) -> None:
        self._update_thresholds(settings)
        csv_tags: list[labels_util.TagMeta] = []
        csv_path = labels_util.discover_labels_csv(settings.tagger.model_path, settings.tagger.tags_csv)
        if csv_path:
            try:
                csv_tags = labels_util.load_selected_tags(csv_path)
            except FileNotFoundError:
                logger.warning("Selected tags CSV not found at %s", csv_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to parse selected tags CSV %s: %s", csv_path, exc)
        tags: list[labels_util.TagMeta] = list(csv_tags)
        seen_names = {tag.name.lower() for tag in tags}
        db_tags: list[str] = []

        # 作品名(IP)ごとの人気度を、紐づくキャラのcountで集計
        ip_counts: dict[str, int] = {}
        for meta in csv_tags:
            if meta.ips:
                base = int(meta.count or 0)
                for ip in meta.ips:
                    ip_counts[ip] = ip_counts.get(ip, 0) + base

        # すべてのIP名をcopyright(=3)として追加（既存と重複はスキップ）
        for ip, cnt in ip_counts.items():
            key = ip.strip().lower()
            if not key:
                continue
            if key in seen_names:
                continue
            seen_names.add(key)
            # 3 は copyright カテゴリ（enum があるなら TagCategory.COPYRIGHT.value を使ってOK）
            tags.append(labels_util.TagMeta(name=ip, category=TagCategory.COPYRIGHT.value, count=cnt))

        if self._conn is not None:
            try:
                db_tags = self._view_model.list_tag_names(self._conn)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to load tag names from database: %s", exc)
        for name in db_tags:
            cleaned = name.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            tags.append(labels_util.TagMeta(name=cleaned, category=0, count=0))
        dedup: dict[str, labels_util.TagMeta] = {}
        for tag in tags:
            key = tag.name.lower()
            existing = dedup.get(key)
            if existing is None or int(tag.count or 0) > int(existing.count or 0):
                dedup[key] = tag
        self._all_tags = labels_util.sort_by_popularity(dedup.values())
        self._update_completion_candidates()
        self._refresh_completions()

    def _show_placeholder(self, show: bool) -> None:
        if show:
            self._stack.setCurrentWidget(self._placeholder)
            self._can_load_more = False
        else:
            target = self._table_view if self._table_button.isChecked() else self._grid_view
            self._stack.setCurrentWidget(target)
        self._update_control_states()

    def _prepare_settings_for_index(self, settings: PipelineSettings | None = None) -> PipelineSettings | None:
        """Return usable settings when at least one scan root exists."""

        try:
            effective = settings or self._view_model.load_settings()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load pipeline settings")
            QMessageBox.critical(
                self,
                "Indexing",
                f"Failed to load pipeline settings:\n{exc}",
            )
            return None

        raw_roots = [Path(root).expanduser() for root in effective.roots if root]
        valid_roots = [root for root in raw_roots if root.exists()]
        if valid_roots:
            return effective

        if raw_roots:
            configured = "\n".join(f"• {root}" for root in raw_roots)
            message = (
                "None of the configured scan roots are available.\n"
                "Update Settings → Roots to reference folders that exist.\n\n"
                f"Configured roots:\n{configured}"
            )
        else:
            message = (
                "No scan roots are configured.\nOpen Settings → Roots and add at least one folder before indexing."
            )

        QMessageBox.warning(self, "Configure scan roots", message)
        return None

    def _on_index_now(self) -> None:
        if self._indexing_active:
            return
        settings = self._prepare_settings_for_index()
        if settings is None:
            return
        self._retag_active = False
        self._db_path = self._resolve_db_path()
        # ★ UI 側の長寿命接続を閉じておく（UNSAFE 区間の EXCLUSIVE ロックを邪魔しない）
        try:
            self.prepare_for_database_reset()
        except Exception:
            pass
        task = IndexRunnable(self._view_model, self._db_path, settings=settings)
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
        settings = self._prepare_settings_for_index()
        if settings is None:
            return
        params_list = list(params or [])

        retag_ids: list[int] = []

        def _pre_run() -> dict[str, object]:
            nonlocal retag_ids
            if predicate is None:
                result = self._view_model.retag_all(self._db_path, force=force_all, settings=settings)
            else:
                result = self._view_model.retag_query(self._db_path, predicate, params_list)
            retag_ids = list(result.file_ids)
            logger.info("Retagging prepared: %d files marked (predicate=%s)", result.affected, predicate)
            return {"retagged_marked": result.affected, "retag_file_ids": retag_ids}

        def _runner(
            progress_cb: Callable[[IndexProgress], None],
            is_cancelled: Callable[[], bool],
        ) -> dict[str, object]:
            return self._view_model.run_retag_selection(
                self._db_path,
                retag_ids,
                settings=settings,
                progress_cb=progress_cb,
                is_cancelled=is_cancelled,
            )

        task = IndexRunnable(
            self._view_model,
            self._db_path,
            settings=settings,
            pre_run=_pre_run,
            runner=_runner,
        )
        self._retag_active = True
        self._start_indexing_task(task)

    def _on_retag_all(self) -> None:
        if self._indexing_active:
            return
        answer = QMessageBox.question(
            self,
            "Retag all files",
            ("Retagging the entire library may take a long time.\nDo you want to continue?"),
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
        logger.info("Retagging current search results: WHERE=%s, PARAMS=%s", self._current_where, self._current_params)
        self._run_retag(
            predicate=self._current_where,
            params=list(self._current_params),
            force_all=False,
        )

    def _on_refresh_clicked(self) -> None:
        if self._refresh_active or self._indexing_active:
            return
        folders = self._determine_refresh_folder()
        if folders is None:
            self._show_toast("Select a file or run a search to choose a folder.")
            return
        modifiers = QApplication.keyboardModifiers()
        hard_delete = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        self._start_refresh_task(folders, hard_delete=hard_delete)

    def _start_refresh_task(self, folders: Sequence[Path], *, hard_delete: bool = False) -> None:
        folder_list = [Path(folder) for folder in folders]
        if not folder_list:
            self._show_toast("No folders to refresh.")
            return

        settings = self._view_model.load_settings()
        self._db_path = self._resolve_db_path()
        self._refresh_active = True
        self._retag_active = False
        self._active_refresh_folder = list(folder_list)

        def _runner(
            progress_cb: Callable[[IndexProgress], None],
            is_cancelled: Callable[[], bool],
        ) -> dict[str, object]:
            return self._view_model.refresh_roots(
                folder_list,
                settings=settings,
                recursive=True,
                hard_delete_missing=hard_delete,
                progress_cb=progress_cb,
                is_cancelled=is_cancelled,
            )

        mode_hint = " (hard delete)" if hard_delete else ""
        self._status_label.setText(f"Refreshing…{mode_hint}")
        self._update_control_states()

        db_path = self._db_path if self._db_path is not None else self._view_model.db_path
        task = IndexRunnable(
            self._view_model,
            db_path,
            settings=settings,
            runner=_runner,
        )
        self._start_indexing_task(task)

    def _determine_refresh_folder(self) -> Sequence[Path] | None:
        settings = self._view_model.load_settings()
        roots = [Path(r).expanduser() for r in (settings.roots or []) if r]
        logger.info("No selection; refreshing all configured roots (%d): %s", len(roots), roots[:3])
        return roots

    def _folder_for_row(self, row: int) -> Path | None:
        if not (0 <= row < len(self._results_cache)):
            return None
        raw_path = str(self._results_cache[row].get("path", ""))
        if not raw_path:
            return None
        base = Path(raw_path).parent
        try:
            return base.resolve(strict=False)
        except OSError:
            return base.absolute()

    def _start_indexing_task(self, task: IndexRunnable) -> None:
        # UIの長寿命接続を閉じて静穏化
        from db.connection import quiesced

        self.prepare_for_database_reset()
        self._enter_quiesce()
        self._quiesce_guard = quiesced()
        self._quiesce_guard.__enter__()

        self._current_index_task = task
        task.signals.progress.connect(self._handle_index_progress)
        task.signals.finished.connect(self._handle_index_finished)
        task.signals.error.connect(self._handle_index_failed)
        self._handle_index_started()
        self._progress_dialog = self._create_progress_dialog()
        self._index_pool.start(task)

    def _enter_quiesce(self) -> None:
        self._release_quiesce()

    def _release_quiesce(self) -> None:
        guard = self._quiesce_guard
        self._quiesce_guard = nullcontext()
        try:
            guard.__exit__(None, None, None)
        except Exception:
            logger.exception("end_quiesce() failed in UI")

    def _create_progress_dialog(self) -> QProgressDialog:
        dialog = QProgressDialog("Preparing…", "Cancel", 0, 0, self)
        if self._refresh_active:
            title = "Refreshing"
        elif self._retag_active:
            title = "Retagging"
        else:
            title = "Indexing"
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoReset(False)
        dialog.setAutoClose(False)
        dialog.canceled.connect(self._cancel_indexing)

        # ★ カスタムラベル（中間省略）を組み込む
        lbl = _ElidingLabel("Preparing…", parent=dialog)
        lbl.set_full_text("Preparing…")
        dialog.setLabel(lbl)
        self._progress_label = lbl

        # ★ 幅の暴れを抑えるため最低幅を決めておく
        dialog.setMinimumWidth(560)

        dialog.show()
        return dialog

    def _cancel_indexing(self) -> None:
        if self._current_index_task is not None:
            self._current_index_task.cancel()
        self._status_label.setText(
            index_cancel_status(refresh_active=self._refresh_active, retag_active=self._retag_active)
        )
        if self._progress_dialog is not None:
            if self._progress_label is not None:
                self._progress_label.set_full_text("Cancelling…")
            else:
                self._progress_dialog.setLabelText("Cancelling…")

    def _handle_index_progress(self, done: int, total: int, label: str) -> None:
        dlg = self._progress_dialog
        if dlg is None:
            return

        try:
            if total < 0:
                dlg.setRange(0, 0)
                if self._progress_label is not None:
                    self._progress_label.set_full_text(label)
                else:
                    dlg.setLabelText(label)
                return

            maximum = max(total, 0)
            # total==0 のとき min(done, total) が常に0になるので、UI的に自然な値にする
            value = max(0, min(done, total if total > 0 else done))

            dlg.setRange(0, maximum)
            dlg.setValue(value)

            percent = min(100, (value * 100) // total) if total > 0 else (100 if value else 0)
            dlg.setLabelText(f"{label}: {value}/{total} ({percent}%)")
        except RuntimeError:
            # ダイアログが deleteLater 済み等で C++ 側が死んでいる場合は無視
            pass

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.hide()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        self._current_index_task = None
        self._progress_label = None

    def _on_table_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._table_view)
            self._grid_button.setChecked(False)

    def _on_grid_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._grid_view)
            self._table_button.setChecked(False)

    def _open_stats(self) -> None:
        db_path = self._db_path if self._db_path is not None else self._view_model.db_path

        def _conn_factory() -> sqlite3.Connection:
            return self._view_model.open_connection(db_path)

        dialog = TagStatsDialog(_conn_factory, parent=self, async_load=True)
        dialog.setModal(True)
        dialog.exec()

    @property
    def _offset(self) -> int:
        return self._search_state.offset

    @_offset.setter
    def _offset(self, value: int) -> None:
        self._search_state.offset = max(0, int(value))

    @property
    def _search_busy(self) -> bool:
        return self._search_state.busy

    @_search_busy.setter
    def _search_busy(self, value: bool) -> None:
        self._search_state.busy = bool(value)

    @property
    def _can_load_more(self) -> bool:
        return self._search_state.can_load_more

    @_can_load_more.setter
    def _can_load_more(self, value: bool) -> None:
        self._search_state.can_load_more = bool(value)

    @property
    def _last_search_cancelled(self) -> bool:
        return self._search_state.last_cancelled

    @_last_search_cancelled.setter
    def _last_search_cancelled(self, value: bool) -> None:
        self._search_state.last_cancelled = bool(value)

    def _set_busy(self, busy: bool) -> None:
        self._search_busy = busy
        if busy:
            self._can_load_more = False
        self._update_control_states()

    def _cancel_active_search(self) -> None:
        worker = self._search_worker
        if worker is not None:
            worker.cancel()

    def _on_search_clicked(self) -> None:
        query = self._query_edit.text().strip()
        positive_terms = self._view_model.extract_positive_terms(query) if query else []
        self._highlight_terms = list(positive_terms)
        self._positive_terms = positive_terms
        self._use_relevance = bool(positive_terms)
        if self._conn is not None:
            try:
                self._relevance_thresholds = self._view_model.load_tag_thresholds(self._conn)
            except sqlite3.Error:
                self._relevance_thresholds = {}
        else:
            self._relevance_thresholds = {}
        thresholds = {int(category): float(value) for category, value in (self._tag_thresholds or {}).items()}
        try:
            fragment = self._view_model.translate_query(
                query,
                file_alias="f",
                thresholds=thresholds,
            )
        except ValueError as exc:
            self._status_label.setText(str(exc))
            self._set_busy(False)
            self._search_overlay.hide()
            return

        order_clause = "relevance DESC, f.mtime DESC" if self._use_relevance else "f.mtime DESC"
        terms_text = ", ".join(self._positive_terms)
        self._debug_where.setText(f"WHERE: {fragment.where}\nORDER: {order_clause}")
        self._debug_params.setText(f"Params: {fragment.params}\nRelevance terms: [{terms_text}]")
        self._debug_group.setVisible(bool(fragment.where.strip() and fragment.where.strip() != "1=1"))

        self._current_query = query
        self._current_where = fragment.where
        self._current_params = list(fragment.params)
        self._search_state.begin_query()
        self._status_label.setText("Searching…")
        self._search_overlay.show("Searching… (Esc to cancel)")
        self._set_busy(True)
        self._start_async_search(reset=True)

    def _on_load_more_clicked(self) -> None:
        if not self._current_where or self._search_busy:
            return
        self._status_label.setText("Searching…")
        self._search_overlay.show("Searching… (Esc to cancel)")
        self._set_busy(True)
        self._start_async_search(reset=False)

    def _start_async_search(self, *, reset: bool) -> None:
        if not self._current_where:
            self._status_label.setText("Enter a query to search tags.")
            self._search_overlay.hide()
            self._set_busy(False)
            self._show_placeholder(True)
            return
        if self._db_path is None:
            self._db_path = self._resolve_db_path()
        if self._db_path is None:
            self._status_label.setText("Database path unavailable.")
            self._search_overlay.hide()
            self._set_busy(False)
            return

        self._cancel_active_search()
        generation = self._search_state.begin_worker(reset=reset)

        offset = 0 if reset else max(0, int(self._offset))
        thresholds = self._relevance_thresholds if self._relevance_thresholds else None
        tags_for_relevance = tuple(self._positive_terms) if self._use_relevance else tuple()

        worker = SearchWorker(
            self._db_path,
            self._current_where or "1=1",
            tuple(self._current_params),
            tags_for_relevance=tags_for_relevance,
            thresholds=thresholds,
            order="relevance" if self._use_relevance else "mtime",
            chunk=self._search_chunk_size,
            offset=offset,
            max_rows=self._search_chunk_size,
            chunk_delay=self._search_chunk_delay,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        self._search_worker = worker
        self._search_thread = thread

        worker.chunkReady.connect(lambda rows, g=generation: self._handle_search_chunk(rows, g))
        worker.finished.connect(
            lambda success, cancelled, g=generation: self._handle_search_finished(success, cancelled, g)
        )
        worker.error.connect(lambda message, g=generation: self._handle_search_error(message, g))

        thread.started.connect(worker.run)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _handle_search_chunk(self, rows: list[dict[str, object]], generation: int) -> None:
        if generation != self._search_state.generation:
            return
        reset_results = self._search_state.reset_pending
        if self._search_state.reset_pending:
            self._clear_results_for_new_search()
            self._search_state.reset_pending = False
        if not rows:
            return
        self._append_rows(rows)
        if reset_results:
            self._scroll_results_to_top()
        self._search_state.consume_rows(len(rows), chunk_size=self._search_chunk_size)
        query_label = self._current_query or "*"
        self._status_label.setText(f"Showing {self._offset} result(s) for '{query_label}'")
        self._show_placeholder(False)

    def _handle_search_error(self, message: str, generation: int) -> None:
        self._search_state.discard_generation(generation)
        if generation != self._search_state.generation:
            return
        self._search_overlay.hide()
        self._set_busy(False)
        self._can_load_more = False
        self._status_label.setText(f"Search failed: {message}")
        self._update_control_states()

    def _handle_search_finished(self, success: bool, cancelled: bool, generation: int) -> None:
        was_reset = self._search_state.finish_generation(generation)
        if generation != self._search_state.generation:
            return
        self._search_worker = None
        self._search_thread = None
        self._search_overlay.hide()
        self._set_busy(False)
        self._last_search_cancelled = bool(cancelled)
        if cancelled:
            self._status_label.setText("Search cancelled.")
            self._can_load_more = False
        elif success and not self._search_state.received_any:
            if self._search_state.reset_pending or was_reset:
                self._clear_results_for_new_search()
                self._search_state.reset_pending = False
                self._status_label.setText("No results. Try indexing your library.")
                self._show_placeholder(True)
                self._can_load_more = False
            else:
                query_label = self._current_query or "*"
                self._status_label.setText(f"Showing {self._offset} result(s) for '{query_label}'")
                self._can_load_more = False
        elif success:
            query_label = self._current_query or "*"
            self._status_label.setText(f"Showing {self._offset} result(s) for '{query_label}'")
        else:
            self._can_load_more = False
        self._update_control_states()

    def _clear_results_for_new_search(self) -> None:
        self._results_cache.clear()
        self._pending_thumbs.clear()
        try:
            self._thumb_pool.clear()
        except Exception:
            pass
        self._table_model.removeRows(0, self._table_model.rowCount())
        self._grid_model.removeRows(0, self._grid_model.rowCount())
        self._table_view.viewport().update()
        self._grid_view.viewport().update()

    def _scroll_results_to_top(self) -> None:
        """Reset both result views to the first item after replacing results."""

        self._table_view.scrollToTop()
        self._grid_view.scrollToTop()
        self._table_view.verticalScrollBar().setValue(0)
        self._grid_view.verticalScrollBar().setValue(0)

    def _append_rows(self, rows: Iterable[dict[str, object]]) -> None:
        for record in rows:
            row_index = len(self._results_cache)
            self._results_cache.append(record)
            path_obj = Path(str(record.get("path", "")))
            raw_tags = list(record.get("tags") or record.get("top_tags") or [])
            # tags = self._filter_display_tags(raw_tags)
            tags = _filter_tags_by_threshold(raw_tags)
            tags_text = self._format_tags(tags)

            table_items = [
                QStandardItem(""),
                QStandardItem(path_obj.name),
                QStandardItem(str(path_obj.parent)),
                QStandardItem(self._format_size(record.get("size"))),
                QStandardItem(self._format_dimensions(record.get("width"), record.get("height"))),
                QStandardItem(self._format_mtime(record.get("mtime"))),
                QStandardItem(tags_text),
            ]
            table_items[0].setData(Qt.AlignmentFlag.AlignCenter, Qt.ItemDataRole.TextAlignmentRole)
            table_items[1].setData(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                Qt.ItemDataRole.TextAlignmentRole,
            )
            table_items[2].setData(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                Qt.ItemDataRole.TextAlignmentRole,
            )
            for item in table_items:
                item.setEditable(False)
            table_items[-1].setToolTip(tags_text)
            table_items[-1].setData(tags, int(_TAG_LIST_ROLE))
            self._table_model.appendRow(table_items)
            self._table_view.setRowHeight(row_index, self._THUMB_SIZE + 16)

            grid_item = QStandardItem(self._format_grid_text(path_obj.name, tags))
            grid_item.setEditable(False)
            grid_item.setData(row_index, Qt.ItemDataRole.UserRole)
            grid_item.setSizeHint(QSize(self._THUMB_SIZE + 48, self._THUMB_SIZE + 72))
            grid_item.setToolTip(tags_text)
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
                table_item.setData(Qt.AlignmentFlag.AlignCenter, Qt.ItemDataRole.TextAlignmentRole)
                self._table_view.setRowHeight(row, max(self._THUMB_SIZE + 16, pixmap.height() + 16))
        if row < self._grid_model.rowCount():
            grid_item = self._grid_model.item(row)
            if grid_item is not None:
                grid_item.setData(pixmap, Qt.ItemDataRole.DecorationRole)

    def _on_table_context_menu(self, pos: QPoint) -> None:
        """Show context menu for a table result."""

        index = self._table_view.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        if not (0 <= row < len(self._results_cache)):
            return
        self._table_view.selectRow(row)
        global_pos = self._table_view.viewport().mapToGlobal(pos)
        self._show_result_context_menu(global_pos, row)

    def _on_grid_context_menu(self, pos: QPoint) -> None:
        """Show context menu for a grid result."""

        index = self._grid_view.indexAt(pos)
        if not index.isValid():
            return
        stored_row = index.data(Qt.ItemDataRole.UserRole)
        row = int(stored_row) if stored_row is not None else index.row()
        if not (0 <= row < len(self._results_cache)):
            return
        self._grid_view.setCurrentIndex(index)
        global_pos = self._grid_view.viewport().mapToGlobal(pos)
        self._show_result_context_menu(global_pos, row)

    def _show_result_context_menu(self, global_pos: QPoint, row: int) -> None:
        """Display the context menu with tag copy actions."""

        if not (0 <= row < len(self._results_cache)):
            return
        menu = QMenu(self)
        copy_plain_action = menu.addAction("タグをコピー（スコア無し）")
        copy_scored_action = menu.addAction("タグをコピー（スコア有り）")
        chosen = menu.exec(global_pos)
        if chosen == copy_plain_action:
            self._copy_tags_to_clipboard(row, include_scores=False)
        elif chosen == copy_scored_action:
            self._copy_tags_to_clipboard(row, include_scores=True)

    def _selected_result_rows(self) -> list[int]:
        """Return currently selected search-result rows."""

        if self._stack.currentWidget() is self._grid_view or self._grid_view.hasFocus():
            indexes = self._grid_view.selectionModel().selectedIndexes()
            if not indexes:
                indexes = [self._grid_view.currentIndex()]
            rows = []
            for index in indexes:
                if not index.isValid():
                    continue
                stored_row = index.data(Qt.ItemDataRole.UserRole)
                rows.append(int(stored_row) if stored_row is not None else index.row())
        else:
            selected = self._table_view.selectionModel().selectedRows()
            if not selected:
                index = self._table_view.currentIndex()
                selected = [index] if index.isValid() else []
            rows = [index.row() for index in selected if index.isValid()]
        return sorted({row for row in rows if 0 <= row < len(self._results_cache)})

    def _on_delete_selected_result(self) -> None:
        """Confirm and start deletion for the selected search result."""

        if self._delete_active or self._search_busy or self._indexing_active or self._refresh_active:
            return
        rows = self._selected_result_rows()
        if not rows:
            return
        entries: list[tuple[int, Path]] = []
        for row in rows:
            record = self._results_cache[row]
            file_id = self._coerce_file_id(record.get("id"))
            if file_id is None:
                continue
            entries.append((file_id, Path(str(record.get("path", "")))))
        if not entries:
            QMessageBox.warning(self, "Delete image", "Selected results do not have valid database ids.")
            return
        paths = [path for _, path in entries]

        answer = QMessageBox.question(
            self,
            "Delete image",
            _format_delete_confirmation(paths),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._delete_active = True
        self._status_label.setText(_format_deleting_status(paths))
        runnable = _DeleteResultRunnable(self._view_model, self._db_path, entries=entries)
        total_entries = len(entries)
        runnable.signals.finished.connect(
            lambda successes, failures: self._handle_delete_finished(successes, failures, total_entries)
        )
        self._delete_pool.start(runnable)

    def _handle_delete_finished(
        self,
        successes: list[tuple[int, str]],
        failures: list[tuple[int, str, str]],
        total: int,
    ) -> None:
        """Remove deleted files from the current result models."""

        self._delete_active = False
        removed = self._remove_results_by_file_ids([file_id for file_id, _ in successes])
        query_label = self._current_query or "*"
        self._status_label.setText(_format_delete_success(successes, total, self._offset, query_label))
        if removed and not self._results_cache and not self._can_load_more:
            self._show_placeholder(True)
        if failures:
            message = "\n".join(f"{path}: {reason}" for _, path, reason in failures)
            QMessageBox.warning(self, "Delete image", f"Some images could not be deleted:\n{message}")
        self._update_control_states()

    def _remove_results_by_file_ids(self, file_ids: Sequence[int]) -> bool:
        """Remove result rows matching *file_ids* from table, grid, and cache."""

        id_set = {int(file_id) for file_id in file_ids}
        if not id_set:
            return False
        rows = [
            index
            for index, record in enumerate(self._results_cache)
            if (file_id := self._coerce_file_id(record.get("id"))) is not None and file_id in id_set
        ]
        if not rows:
            return False
        try:
            self._thumb_pool.clear()
        except Exception:
            pass
        self._pending_thumbs.clear()
        next_selection = min(rows[0], max(0, len(self._results_cache) - len(rows) - 1))
        for row in reversed(rows):
            self._results_cache.pop(row)
            self._table_model.removeRow(row)
            self._grid_model.removeRow(row)
        self._offset = max(0, self._offset - len(rows))
        self._sync_grid_row_roles()
        if self._results_cache:
            self._table_view.selectRow(next_selection)
            self._grid_view.setCurrentIndex(self._grid_model.index(next_selection, 0))
        self._table_view.viewport().update()
        self._grid_view.viewport().update()
        return True

    def _sync_grid_row_roles(self) -> None:
        """Keep grid items pointing at their current result-cache row."""

        for row in range(self._grid_model.rowCount()):
            item = self._grid_model.item(row)
            if item is not None:
                item.setData(row, Qt.ItemDataRole.UserRole)

    @staticmethod
    def _coerce_file_id(value: object) -> int | None:
        """Return *value* as an integer file id when possible."""

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _copy_tags_to_clipboard(self, row: int, *, include_scores: bool) -> None:
        """Copy filtered tags for *row* to the clipboard."""

        if not (0 <= row < len(self._results_cache)):
            self._show_toast("コピー可能なタグがありません。")
            return
        record = self._results_cache[row]
        raw_tags = list(record.get("tags") or record.get("top_tags") or [])
        filtered = _filter_tags_by_threshold(raw_tags)
        if not filtered:
            self._show_toast("コピー可能なタグがありません。")
            return
        if include_scores:
            text = ", ".join(f"{name} ({score:.2f})" for name, score, _ in filtered)
            feedback = "タグ（スコア付き）をクリップボードにコピーしました。"
        else:
            text = ", ".join(name for name, _, _ in filtered)
            feedback = "タグをクリップボードにコピーしました。"
        QApplication.clipboard().setText(text)
        self._show_toast(feedback)

    def _on_table_double_clicked(self, index: QModelIndex) -> None:
        self._open_row(index.row())

    def _on_grid_double_clicked(self, index: QModelIndex) -> None:
        stored_row = index.data(Qt.ItemDataRole.UserRole)
        row = int(stored_row) if stored_row is not None else index.row()
        self._open_row(row)

    def _open_row(self, row: int) -> None:
        if 0 <= row < len(self._results_cache):
            path = Path(str(self._results_cache[row].get("path", "")))
            mods = QApplication.keyboardModifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:  # Ctrlでフォルダ、なしでファイルを開く
                self._open_in_explorer(path)
            else:
                self._open_file_with_default_app(path)

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

    def _filter_display_tags(self, tags: Iterable[Sequence[object]]) -> list[TagDisplayEntry]:
        filtered: list[TagDisplayEntry] = []
        for entry in tags:
            if not entry:
                continue
            name = str(entry[0])
            score_value = entry[1] if len(entry) > 1 else None
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                continue
            category_value = entry[2] if len(entry) > 2 else None
            category = self._normalise_category_value(category_value)
            threshold = 0.0
            if category is not None:
                threshold = self._tag_thresholds.get(category, 0.0)
            if score < threshold:
                continue
            filtered.append((name, score, category))
        return filtered

    def _update_thresholds(self, settings: PipelineSettings | None = None) -> None:
        if settings is None:
            settings = self._view_model.load_settings()
        mapping: dict[TagCategory, float] = {}
        threshold_source = getattr(settings.tagger, "thresholds", {}) if settings else {}
        for key, value in (threshold_source or {}).items():
            category = self._normalise_category_value(key)
            if category is None:
                continue
            try:
                mapping[category] = float(value)
            except (TypeError, ValueError):
                continue
        self._tag_thresholds = mapping

    @staticmethod
    def _normalise_category_value(value: object) -> TagCategory | None:
        return _coerce_category(value)

    @staticmethod
    def _format_tags(tags: Iterable[TagDisplayEntry]) -> str:
        parts = [f"{name} ({score:.2f})" for name, score, _ in tags]
        return ", ".join(parts)

    @staticmethod
    def _format_grid_text(name: str, tags: Iterable[TagDisplayEntry]) -> str:
        tag_names = [entry[0] for entry in tags][:2]
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

    def _open_file_with_default_app(self, path: Path) -> None:
        """Open *path* with the OS default application."""

        try:
            if not path.exists():
                self._status_label.setText(f"File not found: {path}")
                return
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
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
        self._table_view.viewport().update()
        self._grid_view.viewport().update()

    def _update_control_states(self) -> None:
        availability = compute_tags_control_availability(
            TagsActivityState(
                indexing_active=self._indexing_active,
                search_busy=self._search_busy,
                refresh_active=self._refresh_active,
                has_current_query=bool(self._current_where),
                can_load_more=self._can_load_more,
            )
        )
        self._search_button.setEnabled(availability.search)
        self._query_edit.setEnabled(availability.query_input)
        self._load_more_button.setEnabled(availability.load_more)
        self._placeholder_button.setEnabled(availability.placeholder)
        self._table_button.setEnabled(availability.table_view)
        self._grid_button.setEnabled(availability.grid_view)
        self._retag_button.setEnabled(availability.retag)
        self._retag_results_action.setEnabled(availability.retag_results)
        self._refresh_button.setEnabled(availability.refresh)
        if hasattr(self, "_copy_button"):
            self._copy_button.setEnabled(availability.copy_results)

    def _handle_index_started(self) -> None:
        self._indexing_active = True
        self._status_label.setText(
            index_started_status(refresh_active=self._refresh_active, retag_active=self._retag_active)
        )
        self._update_control_states()

    def _handle_index_finished(self, stats: dict[str, object]) -> None:
        task = self._current_index_task
        if task is not None:
            try:
                task.signals.progress.disconnect(self._handle_index_progress)
            except TypeError:
                pass

        self._close_progress_dialog()
        try:
            self.restore_connection()
        except Exception:
            pass

        self._indexing_active = False
        plan = plan_index_finished(
            stats,
            refresh_active=self._refresh_active,
            retag_active=self._retag_active,
            active_refresh_folder=self._active_refresh_folder,
            has_current_query=bool(self._current_where),
        )
        self._refresh_active = plan.refresh_active
        self._retag_active = plan.retag_active
        self._active_refresh_folder = plan.active_refresh_folder
        self._status_label.setText(plan.status)
        self._show_toast(plan.toast)
        self._update_control_states()
        if plan.run_search:
            QTimer.singleShot(0, self._on_search_clicked)
        self._release_quiesce()
        self.restore_connection()

    def _handle_index_failed(self, message: str) -> None:
        task = self._current_index_task
        if task is not None:
            try:
                task.signals.progress.disconnect(self._handle_index_progress)
            except TypeError:
                pass

        self._close_progress_dialog()

        self._indexing_active = False
        plan = plan_index_failed(
            message,
            refresh_active=self._refresh_active,
            retag_active=self._retag_active,
            db_display=self._db_display,
            passthrough_message=ONNXRUNTIME_MISSING_MESSAGE,
        )
        self._status_label.setText(plan.status)
        self._show_toast(plan.toast)
        self._refresh_active = plan.refresh_active
        self._retag_active = plan.retag_active
        self._update_control_states()
        self._release_quiesce()

        # ② 次に接続を復旧（ロックなら短いバックオフで何度か再試行）
        self._restore_connection_with_retry()

    def _restore_connection_with_retry(self, attempts: int = 20, delay_ms: int = 150) -> None:
        try:
            self.restore_connection()
            return
        except Exception as e:
            action = connection_retry_action(e, attempts)
            if action == "raise":
                raise
            if action == "give_up":
                self._show_toast("DB reopen failed (locked). Please try again.")
                return

            QTimer.singleShot(delay_ms, lambda: self._restore_connection_with_retry(attempts - 1, delay_ms))

    def _resolve_db_path(self) -> Path:
        if self._conn is None:
            fallback = Path(self._view_model.db_path).expanduser()
            self._db_display = str(fallback)
            return fallback
        db_row = self._conn.execute("PRAGMA database_list").fetchone()
        literal = db_row[2] if db_row else None
        if literal and literal not in {":memory:", ""}:
            path = Path(literal).expanduser()
            self._db_display = str(path)
            return path
        if literal == ":memory:":
            self._db_display = ":memory:"
            return Path(self._view_model.db_path).expanduser()
        fallback = Path(self._view_model.db_path).expanduser()
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


__all__ = ["TagsTab", "extract_completion_token", "replace_completion_token"]
