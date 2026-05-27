"""Indexing, retagging, refresh, and progress handling for TagsTab."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from core.config import PipelineSettings
from core.jobs import JobPriority
from core.pipeline import IndexProgress
from tagger.wd14_onnx import ONNXRUNTIME_MISSING_MESSAGE
from ui.index_lifecycle import (
    index_cancel_status,
    index_progress_state,
    index_started_status,
    plan_index_failed,
    plan_index_finished,
)
from ui.index_tasks import IndexJob
from ui.tags_workers import _ElidingLabel

logger = logging.getLogger(__name__)


class TagsIndexingMixin:
    """Coordinate long-running index and tag maintenance jobs."""

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
        task = IndexJob(self._view_model, self._db_path, settings=settings, name="index")
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

        task = IndexJob(
            self._view_model,
            self._db_path,
            settings=settings,
            pre_run=_pre_run,
            runner=_runner,
            name="retag",
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
        self._status_label.setText(f"Refreshing...{mode_hint}")
        self._update_control_states()

        db_path = self._db_path if self._db_path is not None else self._view_model.db_path
        task = IndexJob(
            self._view_model,
            db_path,
            settings=settings,
            runner=_runner,
            name="refresh",
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

    def _start_indexing_task(self, task: IndexJob) -> None:
        # UIの長寿命接続を閉じて静穏化
        from db.connection import quiesced

        self.prepare_for_database_reset()
        self._enter_quiesce()
        self._quiesce_guard = quiesced()
        self._quiesce_guard.__enter__()

        handle = self._index_jobs.submit_handle(task, priority=JobPriority.FOREGROUND)
        self._current_index_task = handle
        handle.signals.progressState.connect(self._handle_index_progress_state)
        handle.signals.completed.connect(self._handle_index_finished)
        handle.signals.error.connect(lambda exc, _tb: self._handle_index_failed(str(exc)))
        handle.signals.cancelled.connect(lambda: self._handle_index_finished({"cancelled": True, "elapsed_sec": 0.0}))
        self._handle_index_started()
        self._progress_dialog = self._create_progress_dialog()

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
        dialog = QProgressDialog("Preparing...", "Cancel", 0, 0, self)
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
        lbl = _ElidingLabel("Preparing...", parent=dialog)
        lbl.set_full_text("Preparing...")
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
            self._progress_dialog.setCancelButtonText("Cancelling...")
            self._progress_dialog.setRange(0, 0)
            if self._progress_label is not None:
                self._progress_label.set_full_text("Cancelling...")
            else:
                self._progress_dialog.setLabelText("Cancelling...")

    def _handle_index_progress(self, done: int, total: int, label: str) -> None:
        """Handle the legacy progress signal retained for compatibility."""

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

    def _handle_index_progress_state(self, progress: IndexProgress) -> None:
        """Apply display-ready progress state for indexing tasks."""

        dlg = self._progress_dialog
        if dlg is None:
            return
        state = index_progress_state(progress)
        try:
            if state.indeterminate:
                dlg.setRange(0, 0)
            else:
                dlg.setRange(0, state.maximum)
                dlg.setValue(state.value)
            dlg.setWindowTitle(state.title)
            if self._progress_label is not None:
                self._progress_label.set_full_text(state.label)
            else:
                dlg.setLabelText(state.label)
            self._status_label.setText(state.status)
        except RuntimeError:
            pass

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.hide()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        self._current_index_task = None
        self._progress_label = None

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
                task.signals.progressState.disconnect(self._handle_index_progress_state)
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
                task.signals.progressState.disconnect(self._handle_index_progress_state)
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
