"""Database connection and file export helpers for TagsTab."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox

from core.jobs import CallableJob, JobPriority
from ui.index_lifecycle import connection_retry_action
from ui.tag_stats import TagStatsDialog
from ui.tags_workers import _unique_destination


class TagsDatabaseMixin:
    """Handle database lifecycle and result export actions."""

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

        def _run_copy(
            is_cancelled: Callable[[], bool], emit_progress: Callable[[object], None]
        ) -> tuple[str, int, int]:
            with self._view_model.open_connection(self._db_path) as conn:
                paths = self._view_model.iter_paths_for_search(conn, query)
            total_paths = len(paths)
            emit_progress((0, total_paths))
            ok = ng = 0
            for idx, raw_path in enumerate(paths, start=1):
                if is_cancelled():
                    return str(dest), ok, ng
                try:
                    src = Path(raw_path)
                    if src.exists():
                        shutil.copy2(src, _unique_destination(dest, src.name))
                        ok += 1
                    else:
                        ng += 1
                except Exception:
                    ng += 1
                emit_progress((idx, total_paths))
            return str(dest), ok, ng

        job = CallableJob(_run_copy, name="copy-results")
        handle = self._file_jobs.submit_handle(job, priority=JobPriority.BACKGROUND)
        handle.signals.progressState.connect(
            lambda payload: self._status_label.setText(f"Copying... {payload[0]}/{payload[1]}")
        )

        def _copy_error(exc: Exception, _tb: str) -> None:
            QMessageBox.critical(self, "Copy results", str(exc))
            self._copy_button.setEnabled(True)

        handle.signals.error.connect(_copy_error)

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

        handle.signals.completed.connect(lambda payload: _done(*payload))
        self._copy_button.setEnabled(False)

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

    def _open_stats(self) -> None:
        db_path = self._db_path if self._db_path is not None else self._view_model.db_path

        def _conn_factory() -> sqlite3.Connection:
            return self._view_model.open_connection(db_path)

        dialog = TagStatsDialog(_conn_factory, parent=self, async_load=True)
        dialog.setModal(True)
        dialog.exec()

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
