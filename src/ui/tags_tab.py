"""UI for tag-based search in kobato-eyes."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtCore import QModelIndex, QSize, Qt, QThread, QThreadPool, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCompleter,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.jobs import JobHandle, JobManager
from core.pipeline import run_index_once
from db.connection import get_conn
from tagger import labels_util
from tagger.base import TagCategory
from ui.file_actions import trash_path
from ui.result_delegates import GridThumbDelegate, HoverAwareDelegate, HoverRowTableView
from ui.result_delegates import HighlightDelegate as _HighlightDelegate
from ui.result_delegates import WrappingItemDelegate as _WrappingItemDelegate
from ui.search_worker import SearchWorker
from ui.tag_rendering import _SCORE_COLOR
from ui.tag_rendering import filter_tags_by_threshold as _filter_tags_by_threshold
from ui.tags_autocomplete import TagListModel, TagsAutocompleteMixin, extract_completion_token, replace_completion_token
from ui.tags_db import TagsDatabaseMixin
from ui.tags_indexing import TagsIndexingMixin
from ui.tags_results import TagsResultsMixin
from ui.tags_search import TagsSearchMixin
from ui.tags_workers import _ElidingLabel
from ui.thumbnail_tasks import ThumbnailSignal as _ThumbnailSignal
from ui.viewmodels import TagsSearchState, TagsViewModel
from ui.widgets.spinner_overlay import SpinnerOverlay

logger = logging.getLogger(__name__)


class TagsTab(
    TagsAutocompleteMixin,
    TagsDatabaseMixin,
    TagsIndexingMixin,
    TagsSearchMixin,
    TagsResultsMixin,
    QWidget,
):
    """Provide a search bar and tabular or grid results for tag queries."""

    TagListModel = TagListModel

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
        self._query_edit.setPlaceholderText("Search tags...")
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

        self._autocomplete_event_filters_installed = False
        self._install_autocomplete_event_filters()

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
        self._retag_button.setText("Retag...")
        self._retag_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._retag_button.setMenu(self._retag_menu)
        self._refresh_button = QPushButton("🔄 Refresh", self)
        self._refresh_button.setToolTip("Scan & tag untagged in this folder (Shift+Click = hard delete missing)")
        # 検索バーのボタン群を並べているあたり（_refresh_button のすぐ右あたりが見栄え良い）
        self._copy_button = QPushButton("Copy results...", self)
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
        self._table_view = HoverRowTableView(self)
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
        self._table_view.setItemDelegate(HoverAwareDelegate(self._table_view))
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
        self._current_params: list[object] = []
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
        self._pending_thumbs: set[tuple[int, int]] = set()

        self._index_jobs = JobManager(max_workers=1, parent=self)
        self._indexing_active = False
        self._retag_active = False
        self._refresh_active = False
        self._delete_active = False
        self._progress_dialog: QProgressDialog | None = None
        self._current_index_task: JobHandle | None = None
        self._current_delete_task: JobHandle | None = None
        self._active_refresh_folder: Sequence[Path] | None = None
        self._quiesce_guard: AbstractContextManager[None] = nullcontext()
        self._file_jobs = JobManager(max_workers=1, parent=self)

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
        self._suppress_return_once = False  # Enter誤発火抑止フラグ
        self._progress_label: _ElidingLabel | None = None

        self._on_debug_toggled(False)
        self._show_placeholder(True)
        self._update_control_states()
        self.destroyed.connect(lambda _obj=None: self._remove_autocomplete_event_filters())
        QTimer.singleShot(0, self._initialise_autocomplete)
        QTimer.singleShot(0, self._bootstrap_results_if_any)

    def _install_autocomplete_event_filters(self) -> None:
        """Install completion event filters only on widgets owned by this tab."""

        if self._autocomplete_event_filters_installed:
            return
        self._query_edit.installEventFilter(self)
        popup = self._completer.popup()
        popup.installEventFilter(self)
        popup.viewport().installEventFilter(self)
        self._autocomplete_event_filters_installed = True

    def _remove_autocomplete_event_filters(self) -> None:
        """Remove completion event filters during tab teardown."""

        if not self._autocomplete_event_filters_installed:
            return
        try:
            self._query_edit.removeEventFilter(self)
            popup = self._completer.popup()
            popup.removeEventFilter(self)
            popup.viewport().removeEventFilter(self)
        except RuntimeError:
            logger.debug("Autocomplete event filter removal skipped during Qt teardown", exc_info=True)
        finally:
            self._autocomplete_event_filters_installed = False

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


__all__ = ["TagsTab", "_filter_tags_by_threshold", "extract_completion_token", "replace_completion_token", "trash_path"]
