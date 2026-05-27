"""Result model, thumbnail, context menu, and file actions for TagsTab."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PyQt6.QtCore import QModelIndex, QPoint, QSize, Qt
from PyQt6.QtGui import QPixmap, QStandardItem
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox

from core.config import PipelineSettings
from core.jobs import CallableJob, JobPriority
from tagger.base import TagCategory
from ui.tag_rendering import _TAG_LIST_ROLE, TagDisplayEntry
from ui.tag_rendering import coerce_category as _coerce_category
from ui.tag_rendering import filter_tags_by_threshold as _filter_tags_by_threshold
from ui.tags_control_state import TagsActivityState, compute_tags_control_availability
from ui.tags_delete_state import format_delete_confirmation as _format_delete_confirmation
from ui.tags_delete_state import format_delete_failure_reason as _format_delete_failure_reason
from ui.tags_delete_state import format_delete_result_status as _format_delete_result_status
from ui.tags_delete_state import format_deleting_status as _format_deleting_status
from ui.tags_result_actions import build_copy_tags_payload, collect_delete_entries, normalize_selected_rows
from ui.tags_result_actions import result_row_from_stored as _result_row_from_stored
from ui.tags_result_state import coerce_file_id as _coerce_file_id
from ui.tags_result_state import coerce_result_path as _coerce_result_path
from ui.tags_result_state import plan_result_removal, should_queue_missing_thumbnail, thumbnail_matches_result
from ui.tags_workers import _DeleteResultRunnable
from ui.thumbnail_tasks import ThumbnailTask as _ThumbnailTask


class TagsResultsMixin:
    """Maintain result views and result-scoped actions."""

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

            file_id = self._coerce_file_id(record.get("id"))
            if path_obj.exists() and file_id is not None:
                self._queue_thumbnail(row_index, file_id, path_obj)

    def _queue_thumbnail(self, row: int, file_id: int, path: Path) -> None:
        key = (int(row), int(file_id))
        if key in self._pending_thumbs:
            return
        self._pending_thumbs.add(key)
        task = _ThumbnailTask(row, file_id, path, self._THUMB_SIZE, self._THUMB_SIZE, self._thumb_signal)
        self._thumb_pool.start(task)

    def _apply_thumbnail(self, row: int, file_id: int, pixmap: QPixmap) -> None:
        self._pending_thumbs.discard((int(row), int(file_id)))
        if not self._thumbnail_matches_current_result(row, file_id):
            return
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

    def _thumbnail_matches_current_result(self, row: int, file_id: int) -> bool:
        """Return whether a thumbnail result still belongs to the visible row."""

        return thumbnail_matches_result(self._results_cache, row=row, file_id=file_id)

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
        row = _result_row_from_stored(
            index.row(), index.data(Qt.ItemDataRole.UserRole), result_count=len(self._results_cache)
        )
        if row is None:
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
                rows.append(stored_row if stored_row is not None else index.row())
        else:
            selected = self._table_view.selectionModel().selectedRows()
            if not selected:
                index = self._table_view.currentIndex()
                selected = [index] if index.isValid() else []
            rows = [index.row() for index in selected if index.isValid()]
        return normalize_selected_rows(rows, result_count=len(self._results_cache))

    def _on_delete_selected_result(self) -> None:
        """Confirm and start deletion for the selected search result."""

        if self._delete_active or self._search_busy or self._indexing_active or self._refresh_active:
            return
        rows = self._selected_result_rows()
        if not rows:
            return
        entries = collect_delete_entries(self._results_cache, rows)
        if not entries:
            QMessageBox.warning(self, "Delete image", "Selected results do not have valid database ids or file paths.")
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
        self._update_control_states()
        total_entries = len(entries)

        def _run_delete(
            _is_cancelled: Callable[[], bool],
            _emit_progress: Callable[[object], None],
        ) -> tuple[list[tuple[int, str]], list[int], list[tuple[str, int, str, str]]]:
            task = _DeleteResultRunnable(self._view_model, self._db_path, entries=entries)
            result: list[tuple[list[tuple[int, str]], list[int], list[tuple[str, int, str, str]]]] = []
            task.signals.finished.connect(
                lambda removed, updated, failures: result.append((removed, updated, failures))
            )
            task.run()
            return result[0] if result else ([], [], [])

        job = CallableJob(_run_delete, name="delete-results")
        handle = self._file_jobs.submit_handle(job, priority=JobPriority.FOREGROUND)
        handle.signals.completed.connect(
            lambda payload: self._handle_delete_finished(payload[0], payload[1], payload[2], total_entries)
        )
        handle.signals.error.connect(lambda exc, _tb: self._handle_delete_error(str(exc)))
        self._current_delete_task = handle

    def _handle_delete_error(self, message: str) -> None:
        """Restore UI state after an unexpected delete job failure."""

        self._delete_active = False
        self._current_delete_task = None
        self._status_label.setText(f"Delete failed: {message}")
        QMessageBox.warning(self, "Delete image", f"Delete failed:\n{message}")
        self._update_control_states()

    def _handle_delete_finished(
        self,
        removed: list[tuple[int, str]],
        db_updated_ids: list[int],
        failures: list[tuple[str, int, str, str]],
        total: int,
    ) -> None:
        """Remove deleted files from the current result models."""

        self._delete_active = False
        self._current_delete_task = None
        removed_from_view = self._remove_results_by_file_ids(
            [file_id for file_id, _ in removed],
            offset_file_ids=db_updated_ids,
        )
        query_label = self._current_query or "*"
        self._status_label.setText(
            _format_delete_result_status(removed, failures, total, len(self._results_cache), query_label)
        )
        if removed_from_view and not self._results_cache and not self._can_load_more:
            self._show_placeholder(True)
        if failures:
            message = "\n".join(
                f"{path}: {_format_delete_failure_reason(kind, reason)}" for kind, _, path, reason in failures
            )
            QMessageBox.warning(self, "Delete image", f"Some images could not be deleted:\n{message}")
        self._update_control_states()

    def _remove_results_by_file_ids(self, file_ids: Sequence[int], *, offset_file_ids: Sequence[int]) -> bool:
        """Remove result rows matching *file_ids* from table, grid, and cache."""

        plan = plan_result_removal(self._results_cache, file_ids, offset_file_ids=offset_file_ids)
        if plan is None:
            return False
        try:
            self._thumb_pool.clear()
        except Exception:
            pass
        self._pending_thumbs.clear()
        for row in reversed(plan.rows):
            self._results_cache.pop(row)
            self._table_model.removeRow(row)
            self._grid_model.removeRow(row)
        self._offset = max(0, self._offset - plan.offset_removed)
        self._sync_grid_row_roles()
        if self._results_cache:
            self._table_view.selectRow(plan.next_selection)
            self._grid_view.setCurrentIndex(self._grid_model.index(plan.next_selection, 0))
        self._requeue_missing_thumbnails()
        self._table_view.viewport().update()
        self._grid_view.viewport().update()
        return True

    def _requeue_missing_thumbnails(self) -> None:
        """Queue thumbnails for visible result rows that lost pending work."""

        for row, record in enumerate(self._results_cache):
            work = should_queue_missing_thumbnail(record, has_thumbnail=self._row_has_thumbnail(row))
            if work is None:
                continue
            file_id, path = work
            self._queue_thumbnail(row, file_id, path)

    def _row_has_thumbnail(self, row: int) -> bool:
        """Return whether a table or grid item already has a thumbnail."""

        table_item = self._table_model.item(row, 0) if row < self._table_model.rowCount() else None
        if table_item is not None and table_item.data(Qt.ItemDataRole.DecorationRole) is not None:
            return True
        grid_item = self._grid_model.item(row) if row < self._grid_model.rowCount() else None
        return bool(grid_item is not None and grid_item.data(Qt.ItemDataRole.DecorationRole) is not None)

    def _sync_grid_row_roles(self) -> None:
        """Keep grid items pointing at their current result-cache row."""

        for row in range(self._grid_model.rowCount()):
            item = self._grid_model.item(row)
            if item is not None:
                item.setData(row, Qt.ItemDataRole.UserRole)

    @staticmethod
    def _coerce_file_id(value: object) -> int | None:
        """Return *value* as an integer file id when possible."""

        return _coerce_file_id(value)

    @staticmethod
    def _coerce_result_path(value: object) -> Path | None:
        """Return *value* as a non-empty result path when possible."""

        return _coerce_result_path(value)

    def _copy_tags_to_clipboard(self, row: int, *, include_scores: bool) -> None:
        """Copy filtered tags for *row* to the clipboard."""

        if not (0 <= row < len(self._results_cache)):
            self._show_toast("コピー可能なタグがありません。")
            return
        record = self._results_cache[row]
        raw_tags = list(record.get("tags") or record.get("top_tags") or [])
        payload = build_copy_tags_payload(raw_tags, include_scores=include_scores)
        if payload is None:
            self._show_toast("コピー可能なタグがありません。")
            return
        QApplication.clipboard().setText(payload.text)
        self._show_toast(payload.feedback)

    def _on_table_double_clicked(self, index: QModelIndex) -> None:
        self._open_row(index.row())

    def _on_grid_double_clicked(self, index: QModelIndex) -> None:
        row = _result_row_from_stored(
            index.row(), index.data(Qt.ItemDataRole.UserRole), result_count=len(self._results_cache)
        )
        if row is not None:
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

    def _on_table_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._table_view)
            self._grid_button.setChecked(False)

    def _on_grid_toggled(self, checked: bool) -> None:
        if checked:
            self._stack.setCurrentWidget(self._grid_view)
            self._table_button.setChecked(False)

    def _update_control_states(self) -> None:
        availability = compute_tags_control_availability(
            TagsActivityState(
                indexing_active=self._indexing_active,
                search_busy=self._search_busy,
                refresh_active=self._refresh_active,
                has_current_query=bool(self._current_where),
                can_load_more=self._can_load_more,
                delete_active=self._delete_active,
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
