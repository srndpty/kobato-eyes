"""UI for tag-based search in kobato-eyes."""

from __future__ import annotations

import html
import itertools
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PyQt6.QtCore import (
    QAbstractListModel,
    QEvent,
    QModelIndex,
    QObject,
    QRect,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QKeyEvent,
    QKeySequence,
    QPalette,
    QPixmap,
    QShortcut,
    QStandardItem,
    QStandardItemModel,
    QTextDocument,
)
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
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.config import load_settings
from core.pipeline import IndexProgress, retag_all, retag_query, run_index_once, scan_and_tag
from core.query import extract_positive_tag_terms, translate_query
from core.settings import PipelineSettings
from db.connection import get_conn
from db.repository import _load_tag_thresholds, iter_paths_for_search, list_tag_names, search_files
from tagger import labels_util
from tagger.base import TagCategory
from tagger.wd14_onnx import ONNXRUNTIME_MISSING_MESSAGE
from ui.autocomplete import abbreviate_count, extract_completion_token, replace_completion_token
from ui.tag_stats import TagStatsDialog
from utils.image_io import get_thumbnail
from utils.paths import ensure_dirs, get_db_path
from utils.search_export import make_export_dir

logger = logging.getLogger(__name__)


_CATEGORY_PREFIXES = [f"{category.name.lower()}:" for category in TagCategory]
_RESERVED_COMPLETIONS = ["AND", "OR", "NOT", "category:", *_CATEGORY_PREFIXES]
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
_CATEGORY_KEY_LOOKUP = {
    "0": TagCategory.GENERAL,
    "general": TagCategory.GENERAL,
    "1": TagCategory.CHARACTER,
    "character": TagCategory.CHARACTER,
    "2": TagCategory.RATING,
    "rating": TagCategory.RATING,
    "3": TagCategory.COPYRIGHT,
    "copyright": TagCategory.COPYRIGHT,
    "4": TagCategory.ARTIST,
    "artist": TagCategory.ARTIST,
    "5": TagCategory.META,
    "meta": TagCategory.META,
}


def _category_thresholds() -> dict[TagCategory, float]:
    s = load_settings()
    th = {k.lower(): float(v) for k, v in (s.tagger.thresholds or {}).items()}

    def get(name, default=0.0):
        return th.get(name, default)

    return {
        TagCategory.GENERAL: get("general"),
        TagCategory.CHARACTER: get("character"),
        TagCategory.COPYRIGHT: get("copyright"),
        TagCategory.ARTIST: get("artist"),
        TagCategory.META: get("meta"),
        TagCategory.RATING: get("rating"),
    }


def _filter_tags_by_threshold(tag_rows):
    """tag_rows は (name, score) か (name, score, category) を想定。"""
    out = []
    for row in tag_rows:
        if isinstance(row, dict):
            name = row.get("name")
            score = float(row.get("score", 0.0))
        else:
            if len(row) == 3:
                name, score, cat = row
            elif len(row) == 2:
                name, score = row
            else:
                continue

        if float(score) >= 0.1:  # 0.1以上で固定！ 細かいロングテールタグ問題が鬱陶しいので強制的に解決
            out.append((str(name), float(score)))

    return out


def _rel_luma(color: QColor) -> float:
    """Return the relative luminance of an sRGB color."""

    def _channel(value: int) -> float:
        normalized = value / 255.0
        if normalized <= 0.04045:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    return 0.2126 * _channel(color.red()) + 0.7152 * _channel(color.green()) + 0.0722 * _channel(color.blue())


def _pick_highlight_colors(palette) -> tuple[str, str]:
    """Choose highlight background/foreground colors based on the palette."""

    base = palette.window().color()
    text = palette.text().color()
    is_dark = (_rel_luma(base) < 0.5) or (_rel_luma(text) > 0.7)

    if is_dark:
        background = "#FFD54F"
        foreground = "#000000"
    else:
        background = "#FFF59D"
        foreground = "#000000"
    return background, foreground


_TAG_LIST_ROLE = Qt.ItemDataRole.UserRole + 128


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


# --- バックグラウンドコピー ---------------------------------------------
class _CopyRunnable(QRunnable):
    def __init__(self, db_path: Path, query: str, dest_dir: Path) -> None:
        super().__init__()
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
            with get_conn(self.db_path) as conn:
                paths = list(iter_paths_for_search(conn, self.query))
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


class _HighlightDelegate(QStyledItemDelegate):
    """Render text with highlighted substrings supplied by a provider."""

    def __init__(
        self,
        terms_provider: Callable[[], Iterable[str]],
        parent: QWidget | None = None,
    ) -> None:  # noqa: D401 - Qt signature
        super().__init__(parent)
        self._terms_provider = terms_provider

    @staticmethod
    def _to_html_with_highlight(
        text: str,
        terms: list[str],
        tags: Iterable[tuple[str, float]] | None,
        *,
        bg: str,
        fg: str,
    ) -> str:
        term_set = {term.lower() for term in terms if term}
        if tags and term_set:
            parts: list[str] = []
            for name, score in tags:
                label = f"{name} ({float(score):.2f})"
                escaped = html.escape(label)
                if name.lower() in term_set:
                    parts.append(f'<span style="background-color:{bg}; color:{fg};">{escaped}</span>')
                else:
                    parts.append(escaped)
            return ", ".join(parts)
        return html.escape(text or "")

    def paint(self, painter, option: QStyleOptionViewItem, index):  # noqa: D401 - Qt signature
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        text = str(index.data() or "")
        raw_tags = index.data(int(_TAG_LIST_ROLE))
        tags = list(raw_tags) if raw_tags else []
        terms = list(self._terms_provider() or [])
        background, foreground = _pick_highlight_colors(option.palette)
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(opt.font)
        doc.setHtml(
            self._to_html_with_highlight(
                text,
                terms,
                tags,
                bg=background,
                fg=foreground,
            )
        )
        rect = option.rect
        painter.save()
        painter.translate(rect.topLeft())
        doc.setTextWidth(rect.width())
        doc.drawContents(painter, QRectF(0, 0, rect.width(), rect.height()))
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index):  # noqa: D401 - Qt signature
        text = str(index.data() or "")
        raw_tags = index.data(int(_TAG_LIST_ROLE))
        tags = list(raw_tags) if raw_tags else []
        terms = list(self._terms_provider() or [])
        background, foreground = _pick_highlight_colors(option.palette)
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(option.font)
        doc.setHtml(
            self._to_html_with_highlight(
                text,
                terms,
                tags,
                bg=background,
                fg=foreground,
            )
        )
        available_width = option.rect.width()
        if available_width <= 0 and option.widget is not None:
            available_width = option.widget.width()
        if available_width > 0:
            doc.setTextWidth(available_width)
        size = doc.size().toSize()
        size.setHeight(size.height() + 4)
        return size


class GridThumbDelegate(QStyledItemDelegate):
    """Render thumbnails with captions in the grid view."""

    def __init__(self, thumb_size: int, parent: QWidget | None = None) -> None:  # noqa: D401 - Qt signature
        super().__init__(parent)
        self._thumb = thumb_size

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: D401 - Qt signature
        fm = option.fontMetrics
        text_height = fm.lineSpacing() * 2 + 10
        return QSize(self._thumb + 48, self._thumb + text_height)

    def paint(
        self,
        painter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:  # noqa: D401 - Qt signature
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        style = opt.widget.style() if opt.widget else QApplication.style()
        painter.save()
        style.drawPrimitive(
            QStyle.PrimitiveElement.PE_PanelItemViewItem,
            opt,
            painter,
            opt.widget,
        )
        painter.restore()

        rect = opt.rect
        pix = index.data(Qt.ItemDataRole.DecorationRole)
        icon_bottom = rect.y() + self._thumb
        if isinstance(pix, QPixmap) and not pix.isNull():
            thumb = pix.scaled(
                self._thumb,
                self._thumb,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = rect.x() + (rect.width() - thumb.width()) // 2
            y = rect.y()
            painter.drawPixmap(x, y, thumb)
            icon_bottom = y + thumb.height()

        available_height = max(0, rect.y() + rect.height() - icon_bottom - 4)
        text_rect = QRect(
            rect.x() + 6,
            icon_bottom + 2,
            max(0, rect.width() - 12),
            available_height,
        )
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            return

        palette = opt.palette
        color_group = palette.currentColorGroup()
        is_selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        text_role = QPalette.ColorRole.HighlightedText if is_selected else QPalette.ColorRole.Text
        text_color = palette.color(color_group, text_role)

        base_role = QPalette.ColorRole.Highlight if is_selected else QPalette.ColorRole.Base
        base_color = palette.color(color_group, base_role)
        luminance = 0.299 * base_color.red() + 0.587 * base_color.green() + 0.114 * base_color.blue()
        is_dark_theme = luminance < 128

        if is_dark_theme:
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(base_color)
            painter.drawRoundedRect(text_rect, 4, 4)
            painter.restore()

        fm = opt.fontMetrics
        raw_lines = str(index.data(Qt.ItemDataRole.DisplayRole) or "").splitlines()
        if not raw_lines:
            raw_lines = [""]
        if len(raw_lines) >= 2:
            lines = [raw_lines[0], " ".join(raw_lines[1:])]
        else:
            lines = [raw_lines[0]]
        lines = [fm.elidedText(line, Qt.TextElideMode.ElideRight, text_rect.width()) for line in lines]
        lines = [line for line in lines if line] or [""]

        total_height = fm.lineSpacing() * len(lines)
        y_start = text_rect.y() + max(0, text_rect.height() - total_height)

        painter.save()
        painter.setPen(text_color)
        for i, line in enumerate(lines):
            line_rect = QRect(
                text_rect.x(),
                y_start + i * fm.lineSpacing(),
                text_rect.width(),
                fm.lineSpacing(),
            )
            painter.drawText(
                line_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                line,
            )
        painter.restore()


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

        try:
            self.signals.progress.emit(progress.done, progress.total, label)
        except RuntimeError:
            # ウィジェット/Signalsが破棄済み。静かに無視。
            return

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


class RefreshRunnable(QRunnable):
    """Run ``scan_and_tag`` on a worker thread and forward the resulting summary."""

    class Signals(QObject):
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

    def __init__(
        self,
        folder: Path,
        *,
        recursive: bool = True,
        batch_size: int = 8,
        hard_delete: bool = False,
    ) -> None:
        super().__init__()
        self._folder = Path(folder)
        self._recursive = recursive
        self._batch_size = batch_size
        self._hard_delete = hard_delete
        self.signals = self.Signals()

    def run(self) -> None:  # noqa: D401
        try:
            stats = scan_and_tag(
                self._folder,
                recursive=self._recursive,
                batch_size=self._batch_size,
                skip_hnsw=True,
                hard_delete_missing=self._hard_delete,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Manual refresh failed for %s", self._folder)
            self.signals.error.emit(str(exc))
            return
        stats = dict(stats)
        stats["folder"] = str(self._folder)
        stats["hard_delete"] = self._hard_delete
        self.signals.finished.emit(stats)


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
                return f"{item.name} ({count_text})" if count_text else item.name
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

        def reset_with(self, items: Sequence[labels_util.TagMeta]) -> None:
            self.beginResetModel()
            self._items = list(items)
            self.endResetModel()

    _PAGE_SIZE = 200
    _THUMB_SIZE = 128

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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
        self._table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_view.doubleClicked.connect(self._on_table_double_clicked)
        # self._table_view.activated.connect(self._on_table_double_clicked)
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
        self._grid_view.activated.connect(self._on_grid_double_clicked)
        self._grid_view.setModel(self._grid_model)

        self._highlight_terms: list[str] = []
        self._positive_terms: list[str] = []
        self._relevance_thresholds: dict[int, float] = {}
        self._use_relevance = False
        self._tags_delegate = _HighlightDelegate(lambda: self._highlight_terms, self._table_view)
        tags_col = self._table_model.columnCount() - 1
        if tags_col >= 0:
            self._table_view.setItemDelegateForColumn(tags_col, self._tags_delegate)
        self._grid_delegate = GridThumbDelegate(self._THUMB_SIZE, self._grid_view)
        self._grid_view.setItemDelegate(self._grid_delegate)

        self._stack.addWidget(self._placeholder)
        self._stack.addWidget(self._table_view)
        self._stack.addWidget(self._grid_view)

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
        self._retag_all_action.triggered.connect(self._on_retag_all)
        self._retag_results_action.triggered.connect(self._on_retag_results)
        self._table_button.toggled.connect(self._on_table_toggled)
        self._grid_button.toggled.connect(self._on_grid_toggled)
        self._stats_button.clicked.connect(self._open_stats)
        self._placeholder_button.clicked.connect(self._on_index_now)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)

        self._current_query: Optional[str] = None
        self._current_where: Optional[str] = None
        self._current_params: List[object] = []
        self._offset = 0
        self._results_cache: list[dict[str, object]] = []
        self._tag_thresholds: dict[TagCategory, float] = {}

        ensure_dirs()
        self._db_display = str(get_db_path())
        self._conn: sqlite3.Connection | None = None
        self._open_connection()
        self._db_path = self._resolve_db_path()
        self.destroyed.connect(self._close_connection)
        self._update_thresholds(load_settings())

        self._thumb_pool = QThreadPool(self)
        self._thumb_pool.setMaxThreadCount(min(4, self._thumb_pool.maxThreadCount()))
        self._thumb_signal = _ThumbnailSignal()
        self._thumb_signal.finished.connect(self._apply_thumbnail)
        self._pending_thumbs: set[int] = set()

        self._index_pool = QThreadPool(self)
        self._index_pool.setMaxThreadCount(1)
        self._refresh_pool = QThreadPool(self)
        self._refresh_pool.setMaxThreadCount(1)
        self._search_busy = False
        self._indexing_active = False
        self._retag_active = False
        self._refresh_active = False
        self._can_load_more = False
        self._progress_dialog: QProgressDialog | None = None
        self._current_index_task: IndexRunnable | None = None
        self._current_refresh_task: RefreshRunnable | None = None
        self._active_refresh_folder: Path | None = None

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

        self._query_edit.installEventFilter(self)
        self._suppress_return_once = False  # Enter誤発火抑止フラグ
        self._progress_label: _ElidingLabel | None = None

        self._on_debug_toggled(False)
        self._show_placeholder(True)
        self._update_control_states()
        QTimer.singleShot(0, self._initialise_autocomplete)
        QTimer.singleShot(0, self._bootstrap_results_if_any)

    def _on_copy_results_clicked(self) -> None:
        # 現在の検索が確定していることを確認
        if not self._current_query or not self._current_where:
            QMessageBox.information(self, "Copy results", "Run a search first.")
            return

        query = self._current_query.strip() or "*"

        # 件数の事前確認（UI表示用）
        try:
            with get_conn(self._db_path) as conn:
                total = sum(1 for _ in iter_paths_for_search(conn, query))
        except Exception as e:
            QMessageBox.critical(self, "Copy results", f"Failed to enumerate results:\n{e}")
            return

        if total <= 0:
            QMessageBox.information(self, "Copy results", "No results to copy.")
            return

        dest = make_export_dir(query)
        choice = QMessageBox.question(
            self,
            "Copy results",
            f"Copy {total} file(s) to:\n{dest}\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return

        runnable = _CopyRunnable(self._db_path, query, dest)
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
        if obj is self._query_edit and event.type() == QEvent.Type.KeyPress:
            e: QKeyEvent = event  # type: ignore[assignment]
            # try:
            #     name = obj.objectName()
            # except Exception:
            #     name = obj.__class__.__name__
            # print(f"[filter] obj={name} key={e.key()} popupVisible={self._completer.popup().isVisible()}")

            key = e.key()
            popup = self._completer.popup()
            popup_visible = bool(popup and popup.isVisible())
            # print(f"event:{event}, key:{key}")

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
        settings = load_settings()
        self.reload_autocomplete(settings)

    def _open_connection(self) -> None:
        if self._conn is not None:
            return
        self._conn = get_conn(get_db_path())

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
            self._current_query = "*"
            self._current_where = "1=1"
            self._current_params = []
            self._offset = 0
            self._results_cache.clear()
            self._pending_thumbs.clear()
            self._table_model.removeRows(0, self._table_model.rowCount())
            self._grid_model.removeRows(0, self._grid_model.rowCount())
            self._highlight_terms = []
            self._positive_terms = []
            self._relevance_thresholds = {}
            self._use_relevance = False
            self._debug_where.setText("WHERE: 1=1")
            self._debug_params.setText("Params: []")
            self._debug_group.setVisible(False)
            self._show_placeholder(False)
            self._fetch_results(reset=True)
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
        self.reload_autocomplete(load_settings())

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

    def _refresh_completions(self) -> None:
        if not self._completion_candidates:
            self._tag_model.reset_with([])
            self._hide_completion_popup()
            return
        text = self._query_edit.text()
        cursor_position = self._query_edit.cursorPosition()
        token, start, end = extract_completion_token(text, cursor_position)
        self._current_completion_range = (start, end)
        prefix = token.lower()
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
            self._tag_model.reset_with(limited)
            # ★ ここが肝心：QCompleter にも “このトークン” を prefix として教える
            self._completer.setCompletionPrefix(token)
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
        print(f"base_text:{base_text}, new_text:{new_text}, cursor:{cursor}")
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
        if self._conn is not None:
            try:
                db_tags = list_tag_names(self._conn)
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
            ("Retagging the entire library may take a long time.\n" "Do you want to continue?"),
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

    def _on_refresh_clicked(self) -> None:
        if self._refresh_active or self._indexing_active:
            return
        folder = self._determine_refresh_folder()
        if folder is None:
            self._show_toast("Select a file or run a search to choose a folder.")
            return
        modifiers = QApplication.keyboardModifiers()
        hard_delete = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        self._start_refresh_task(folder, hard_delete=hard_delete)

    def _start_refresh_task(self, folder: Path, *, hard_delete: bool = False) -> None:
        self._refresh_active = True
        self._active_refresh_folder = folder
        task = RefreshRunnable(folder, hard_delete=hard_delete)
        self._current_refresh_task = task
        task.signals.finished.connect(self._handle_refresh_finished)
        task.signals.error.connect(self._handle_refresh_failed)
        mode_hint = " (hard delete)" if hard_delete else ""
        self._status_label.setText(f"Scanning…{mode_hint}")
        self._update_control_states()
        self._refresh_pool.start(task)

    def _determine_refresh_folder(self) -> Path | None:
        folder: Path | None = None
        if self._stack.currentWidget() is self._table_view:
            model = self._table_view.selectionModel()
            if model is not None:
                rows = model.selectedRows()
                if rows:
                    folder = self._folder_for_row(rows[0].row())
        elif self._stack.currentWidget() is self._grid_view:
            model = self._grid_view.selectionModel()
            if model is not None:
                indexes = model.selectedIndexes()
                if indexes:
                    stored_row = indexes[0].data(Qt.ItemDataRole.UserRole)
                    row_index = int(stored_row) if stored_row is not None else indexes[0].row()
                    folder = self._folder_for_row(row_index)
        if folder is None and self._results_cache:
            folder = self._folder_for_row(0)
        return folder

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

    def _handle_refresh_finished(self, stats: dict[str, object]) -> None:
        task = self._current_refresh_task
        if task is not None:
            try:
                task.signals.finished.disconnect(self._handle_refresh_finished)
            except TypeError:
                pass
            try:
                task.signals.error.disconnect(self._handle_refresh_failed)
            except TypeError:
                pass
        self._current_refresh_task = None
        self._refresh_active = False
        folder = self._active_refresh_folder
        self._active_refresh_folder = None
        queued = int(stats.get("queued", 0) or 0)
        tagged = int(stats.get("tagged", 0) or 0)
        elapsed = float(stats.get("elapsed_sec", 0.0) or 0.0)
        missing = int(stats.get("missing", 0) or 0)
        soft_deleted = int(stats.get("soft_deleted", 0) or 0)
        hard_deleted = int(stats.get("hard_deleted", 0) or 0)
        removed_total = soft_deleted + hard_deleted
        if missing <= 0 and removed_total > 0:
            missing = removed_total
        hard_delete = bool(stats.get("hard_delete", False) or hard_deleted)
        removal_label = "hard delete" if hard_delete else "soft delete"
        folder_text = str(folder) if folder is not None else str(stats.get("folder", ""))
        status = (
            f"Refresh complete: {tagged} tagged added, {removed_total} missing removed "
            f"({removal_label}, {elapsed:.2f}s; queued {queued})."
        )
        if folder_text:
            status += f" [{folder_text}]"
        self._status_label.setText(status)
        toast = f"{tagged} tagged added; {removed_total} missing removed"
        if hard_delete:
            toast += " (hard delete)"
        if folder_text:
            toast = f"{toast} {folder_text}"
        self._show_toast(toast)
        self._update_control_states()
        if self._current_where:
            QTimer.singleShot(0, self._on_search_clicked)

    def _handle_refresh_failed(self, message: str) -> None:
        task = self._current_refresh_task
        if task is not None:
            try:
                task.signals.finished.disconnect(self._handle_refresh_finished)
            except TypeError:
                pass
            try:
                task.signals.error.disconnect(self._handle_refresh_failed)
            except TypeError:
                pass
        self._current_refresh_task = None
        self._refresh_active = False
        folder = self._active_refresh_folder
        self._active_refresh_folder = None
        folder_text = f" {folder}" if folder is not None else ""
        error_text = f"Refresh failed{folder_text}: {message}"
        self._status_label.setText(error_text)
        self._show_toast(error_text)
        self._update_control_states()

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
        prefix = "Retagging" if self._retag_active else "Indexing"
        self._status_label.setText(f"{prefix} cancelling…")
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
        db_path = self._db_path if self._db_path is not None else Path(get_db_path())

        def _conn_factory() -> sqlite3.Connection:
            return get_conn(db_path)

        dialog = TagStatsDialog(_conn_factory, parent=self)
        dialog.setModal(True)
        dialog.exec()

    def _set_busy(self, busy: bool) -> None:
        self._search_busy = busy
        if busy:
            self._can_load_more = False
        self._update_control_states()

    def _on_search_clicked(self) -> None:
        query = self._query_edit.text().strip()
        positive_terms = extract_positive_tag_terms(query) if query else []
        self._highlight_terms = list(positive_terms)
        self._positive_terms = positive_terms
        self._use_relevance = bool(positive_terms)
        if self._conn is not None:
            try:
                self._relevance_thresholds = _load_tag_thresholds(self._conn)
            except sqlite3.Error:
                self._relevance_thresholds = {}
        else:
            self._relevance_thresholds = {}
        self._set_busy(True)
        thresholds = {int(category): float(value) for category, value in (self._tag_thresholds or {}).items()}
        try:
            fragment = translate_query(
                query,
                file_alias="f",
                thresholds=thresholds,
            )
        except ValueError as exc:
            self._status_label.setText(str(exc))
            self._set_busy(False)
            self._table_view.viewport().update()
            self._grid_view.viewport().update()
            return

        order_clause = "relevance DESC, f.mtime DESC" if self._use_relevance else "f.mtime DESC"
        terms_text = ", ".join(self._positive_terms)
        self._debug_where.setText(f"WHERE: {fragment.where}\nORDER: {order_clause}")
        self._debug_params.setText(f"Params: {fragment.params}\nRelevance terms: [{terms_text}]")
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
        self._table_view.viewport().update()
        self._grid_view.viewport().update()

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
        if self._conn is None:
            self._status_label.setText("Database connection unavailable.")
            self._set_busy(False)
            return
        try:
            rows = search_files(
                self._conn,
                self._current_where,
                self._current_params,
                tags_for_relevance=self._positive_terms,
                thresholds=self._relevance_thresholds,
                order="relevance" if self._use_relevance else "mtime",
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

    def _filter_display_tags(self, tags: Iterable[Sequence[object]]) -> list[tuple[str, float]]:
        filtered: list[tuple[str, float]] = []
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
            filtered.append((name, score))
        return filtered

    def _update_thresholds(self, settings: PipelineSettings | None = None) -> None:
        if settings is None:
            settings = load_settings()
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
        if value is None:
            return None
        if isinstance(value, TagCategory):
            return TagCategory(int(value))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return TagCategory(int(value))
            except (ValueError, TypeError):
                return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        category = _CATEGORY_KEY_LOOKUP.get(lowered)
        if category is not None:
            return category
        try:
            return TagCategory(int(float(text)))
        except (ValueError, TypeError):
            return None

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
        search_enabled = not self._search_busy and not self._indexing_active
        input_enabled = not self._indexing_active and not self._search_busy
        self._search_button.setEnabled(search_enabled)
        self._query_edit.setEnabled(input_enabled)
        self._load_more_button.setEnabled(self._can_load_more and not self._indexing_active and not self._search_busy)
        self._placeholder_button.setEnabled(not self._indexing_active)
        self._table_button.setEnabled(not self._indexing_active)
        self._grid_button.setEnabled(not self._indexing_active)
        retag_enabled = not self._indexing_active and not self._search_busy
        self._retag_button.setEnabled(retag_enabled)
        self._retag_results_action.setEnabled(bool(self._current_where) and retag_enabled)
        refresh_enabled = not self._refresh_active and not self._indexing_active and not self._search_busy
        self._refresh_button.setEnabled(refresh_enabled)
        # ★ 検索済み & 処理中でない時だけコピーを有効化
        copy_enabled = bool(self._current_where) and not (
            self._search_busy or self._indexing_active or self._refresh_active
        )
        if hasattr(self, "_copy_button"):
            self._copy_button.setEnabled(copy_enabled)

    def _handle_index_started(self) -> None:
        self._indexing_active = True
        if self._retag_active:
            self._status_label.setText("Retagging…")
        else:
            self._status_label.setText("Indexing…")
        self._update_control_states()

    def _handle_index_finished(self, stats: dict[str, object]) -> None:
        task = self._current_index_task
        if task is not None:
            try:
                task.signals.progress.disconnect(self._handle_index_progress)
            except TypeError:
                pass

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
        task = self._current_index_task
        if task is not None:
            try:
                task.signals.progress.disconnect(self._handle_index_progress)
            except TypeError:
                pass

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
        if self._conn is None:
            fallback = Path(get_db_path()).expanduser()
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


__all__ = ["TagsTab", "extract_completion_token", "replace_completion_token"]
