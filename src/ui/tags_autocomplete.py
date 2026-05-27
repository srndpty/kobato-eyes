"""Autocomplete behavior for the tag search tab."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from PyQt6.QtCore import QAbstractListModel, QEvent, QModelIndex, QObject, Qt, QTimer
from PyQt6.QtGui import QKeyEvent

from core.config import PipelineSettings
from tagger import labels_util
from tagger.base import TagCategory
from ui.autocomplete import (
    abbreviate_count,
    completion_search_prefix,
    extract_completion_token,
    replace_completion_token,
)

logger = logging.getLogger(__name__)

_CATEGORY_PREFIXES = [f"{category.name.lower()}:" for category in TagCategory]
_RESERVED_COMPLETIONS = ["category:", *_CATEGORY_PREFIXES]


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


class TagsAutocompleteMixin:
    """Provide query completion and key handling for TagsTab."""

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
            self._status_label.setText("Searching...")
            self._search_overlay.show("Loading latest... (Esc to cancel)")
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
