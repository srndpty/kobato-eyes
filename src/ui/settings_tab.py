"""Prototype settings UI for kobato-eyes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from PyQt6.QtCore import QObject, QRunnable, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import PipelineSettings, TaggerSettings
from core.jobs import CallableJob, JobHandle, JobManager, JobPriority
from tagger.model_inspection import format_inspection, inspect_model
from ui.viewmodels import SettingsViewModel

if TYPE_CHECKING:
    from core.pipeline import ProcessingPipeline
    from ui.tags_tab import TagsTab


logger = logging.getLogger(__name__)


class _ModelInspectionSignals(QObject):
    """Signals emitted by the background model inspection task."""

    finished = pyqtSignal(int, str, bool)


class _ModelInspectionRunnable(QRunnable):
    """Run model inspection outside the Qt UI thread."""

    def __init__(
        self,
        generation: int,
        *,
        tagger_name: str,
        model_path: str | None,
        tags_csv: str | None,
        provider_loader: Callable[[], Iterable[str]],
    ) -> None:
        super().__init__()
        self._generation = generation
        self._tagger_name = tagger_name
        self._model_path = model_path
        self._tags_csv = tags_csv
        self._provider_loader = provider_loader
        self.signals = _ModelInspectionSignals()

    def run(self) -> None:
        """Inspect model settings and emit a formatted summary."""

        try:
            inspection = inspect_model(
                tagger_name=self._tagger_name,
                model_path=self._model_path,
                tags_csv=self._tags_csv,
                provider_loader=self._provider_loader,
            )
            message = format_inspection(inspection)
            ok = inspection.ok
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            logger.exception("Failed to inspect tagger model")
            message = f"Model status: Error\nError: {exc}"
            ok = False
        self.signals.finished.emit(self._generation, message, ok)


def _format_size(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _database_size(db_path: Path) -> int:
    total = 0
    for candidate in (
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    ):
        try:
            total += candidate.stat().st_size
        except OSError:
            continue
    return total


class ResetDatabaseDialog(QDialog):
    """Confirm destructive database resets with optional backups."""

    def __init__(self, db_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reset database")
        self.setModal(True)

        description = QLabel(self)
        description.setWordWrap(True)
        size_text = _format_size(_database_size(db_path))
        description.setText(
            (
                "This will delete the current database and recreate an empty schema.\n\n"
                f"Path: {db_path}\n"
                f"Current size (including WAL/SHM): {size_text}\n\n"
                "Existing search results and tags will be lost."
            )
        )

        self._backup_check = QCheckBox("Backup .db / -wal / -shm before deleting (recommended)", self)
        self._backup_check.setChecked(True)

        self._rescan_check = QCheckBox("Start indexing immediately after reset", self)
        self._rescan_check.setChecked(True)

        confirm_label = QLabel("Type RESET to confirm:", self)
        self._confirm_edit = QLineEdit(self)
        self._confirm_edit.setPlaceholderText("RESET")
        self._confirm_edit.textChanged.connect(self._update_button_state)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
            self,
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        ok_button = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addWidget(description)
        layout.addWidget(self._backup_check)
        layout.addWidget(self._rescan_check)
        layout.addWidget(confirm_label)
        layout.addWidget(self._confirm_edit)
        layout.addWidget(self._button_box)

    def _update_button_state(self, text: str) -> None:
        ok_button = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(text.strip() == "RESET")

    @property
    def backup_enabled(self) -> bool:
        return self._backup_check.isChecked()

    @property
    def start_index_enabled(self) -> bool:
        return self._rescan_check.isChecked()


class SettingsTab(QWidget):
    """Provide minimal controls for pipeline configuration."""

    settings_applied = pyqtSignal(PipelineSettings)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        view_model: SettingsViewModel | None = None,
    ) -> None:
        super().__init__(parent)
        self._view_model = view_model or SettingsViewModel(self)
        self._view_model.settings_applied.connect(self.settings_applied.emit)
        self._inspection_jobs = JobManager(max_workers=1, parent=self)
        self._inspection_task: JobHandle | None = None
        self._inspection_generation = 0
        self._inspection_timer = QTimer(self)
        self._inspection_timer.setSingleShot(True)
        self._inspection_timer.setInterval(350)
        self._inspection_timer.timeout.connect(self._start_model_inspection)
        self._loading_settings = False
        self._dirty = False

        self._roots_edit = QPlainTextEdit(self)
        self._roots_edit.setPlaceholderText("One path per line")
        self._roots_edit.textChanged.connect(self._mark_dirty)
        self._excluded_edit = QPlainTextEdit(self)
        self._excluded_edit.setPlaceholderText("Paths to ignore")
        self._excluded_edit.textChanged.connect(self._mark_dirty)

        self._batch_size_spin = QSpinBox(self)
        self._batch_size_spin.setRange(1, 512)
        self._batch_size_spin.setValue(8)
        self._batch_size_spin.setToolTip("Number of images processed per inference batch")
        self._batch_size_spin.valueChanged.connect(self._mark_dirty)

        self._prefetch_depth_spin = QSpinBox(self)
        self._prefetch_depth_spin.setRange(1, 64)
        self._prefetch_depth_spin.setValue(4)
        self._prefetch_depth_spin.setToolTip("Number of batches prepared ahead of the tagger")
        self._prefetch_depth_spin.valueChanged.connect(self._mark_dirty)

        self._tagger_input_cache_check = QCheckBox("Cache prepared PNG tagger inputs", self)
        self._tagger_input_cache_check.setToolTip(
            "Store resized PNG inputs in the app cache to speed up repeated tagging"
        )
        self._tagger_input_cache_check.stateChanged.connect(self._mark_dirty)

        self._device_combo = QComboBox(self)
        self._device_combo.addItem("Auto", "auto")
        self._device_combo.addItem("TensorRT then CUDA", "tensorrt")
        self._device_combo.addItem("CUDA only", "cuda")
        self._device_combo.addItem("CPU only", "cpu")
        self._device_combo.currentIndexChanged.connect(self._mark_dirty)

        self._tagger_combo = QComboBox(self)
        self._tagger_combo.addItems(["dummy", "wd14-onnx"])
        self._tagger_combo.currentTextChanged.connect(self._update_tagger_inputs)
        self._tagger_combo.currentTextChanged.connect(self._mark_dirty)

        self._tagger_model_edit = QLineEdit(self)
        self._tagger_model_edit.setPlaceholderText("Path to WD14 ONNX model")
        self._tagger_model_edit.textChanged.connect(self._schedule_model_inspection)
        self._tagger_model_edit.textChanged.connect(self._mark_dirty)
        self._tagger_model_button = QPushButton("Browse...", self)
        self._tagger_model_button.clicked.connect(self._on_browse_model)
        tagger_model_row = QWidget(self)
        tagger_layout = QHBoxLayout(tagger_model_row)
        tagger_layout.setContentsMargins(0, 0, 0, 0)
        tagger_layout.addWidget(self._tagger_model_edit)
        tagger_layout.addWidget(self._tagger_model_button)

        self._model_info_edit = QPlainTextEdit(self)
        self._model_info_edit.setReadOnly(True)
        self._model_info_edit.setMinimumHeight(118)
        self._model_info_edit.setMaximumHeight(180)
        self._model_info_edit.setPlaceholderText("Model diagnostics will appear here.")

        self._status_label = QLabel("Settings are up to date.", self)
        self._apply_button = QPushButton("Apply", self)
        self._apply_button.clicked.connect(self._emit_settings)

        self._reset_button = QPushButton("Reset database...", self)
        self._reset_button.clicked.connect(self._on_reset_database)

        form = QFormLayout()
        form.addRow(QLabel("Roots"), self._roots_edit)
        form.addRow(QLabel("Excluded"), self._excluded_edit)
        form.addRow("Batch size", self._batch_size_spin)
        form.addRow("Prefetch depth", self._prefetch_depth_spin)
        form.addRow("Input cache", self._tagger_input_cache_check)
        form.addRow("Device", self._device_combo)
        form.addRow("Tagger", self._tagger_combo)
        form.addRow("Model path", tagger_model_row)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch()
        layout.addWidget(QLabel("Model diagnostics", self))
        layout.addWidget(self._model_info_edit)
        buttons = QHBoxLayout()
        buttons.addWidget(self._reset_button)
        buttons.addWidget(self._status_label)
        buttons.addStretch()
        buttons.addWidget(self._apply_button)
        layout.addLayout(buttons)

        self._current_settings: PipelineSettings = self._view_model.current_settings
        self._pipeline: ProcessingPipeline | None = None
        self._tags_tab: TagsTab | None = None
        self.load_settings(self._current_settings)

    def load_settings(self, settings: PipelineSettings) -> None:
        self._loading_settings = True
        try:
            self._current_settings = settings
            self._view_model.set_current_settings(settings)
            self._roots_edit.setPlainText("\n".join(str(path) for path in settings.roots))
            self._excluded_edit.setPlainText("\n".join(str(path) for path in settings.excluded))
            self._batch_size_spin.setValue(settings.batch_size)
            self._prefetch_depth_spin.setValue(settings.prefetch_depth)
            self._tagger_input_cache_check.setChecked(settings.tagger_input_cache)
            device_index = self._device_combo.findData(settings.tagger.device)
            self._device_combo.setCurrentIndex(device_index if device_index >= 0 else 0)

            tagger_index = self._tagger_combo.findText(settings.tagger.name)
            if tagger_index >= 0:
                self._tagger_combo.setCurrentIndex(tagger_index)
            else:
                self._tagger_combo.setCurrentText(settings.tagger.name)
            self._tagger_model_edit.setText(settings.tagger.model_path or "")
            self._update_tagger_inputs(self._tagger_combo.currentText())
        finally:
            self._loading_settings = False
            self._set_dirty(False)

    def _emit_settings(self) -> None:
        settings = self._collect_settings()
        self._current_settings = settings
        self._view_model.apply_settings(settings)
        self._set_dirty(False)

    def _collect_settings(self) -> PipelineSettings:
        current = self._current_settings or PipelineSettings()
        previous_tagger = current.tagger if current else TaggerSettings()
        tagger_name = self._tagger_combo.currentText()
        is_wd14 = tagger_name.lower() == "wd14-onnx"
        model_path_text = self._tagger_model_edit.text().strip()
        model_path = model_path_text if is_wd14 and model_path_text else None
        settings = self._view_model.build_settings(
            roots=[Path(line) for line in self._lines(self._roots_edit) if line],
            excluded=[Path(line) for line in self._lines(self._excluded_edit) if line],
            batch_size=self._batch_size_spin.value(),
            prefetch_depth=self._prefetch_depth_spin.value(),
            tagger_input_cache=self._tagger_input_cache_check.isChecked(),
            tagger_name=tagger_name,
            model_path=model_path,
            device=str(self._device_combo.currentData() or "auto"),
            previous_tagger=previous_tagger,
            previous_settings=current,
        )
        return settings

    @staticmethod
    def _lines(edit: QPlainTextEdit) -> Iterable[str]:
        return (line.strip() for line in edit.toPlainText().splitlines())

    def _mark_dirty(self, *_args: object) -> None:
        if not self._loading_settings:
            self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self._apply_button.setEnabled(dirty)
        if dirty:
            self._status_label.setText("Unapplied changes will be saved when leaving this tab.")
        else:
            self._status_label.setText("Settings are up to date.")

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Auto-apply pending edits when the user leaves the Settings tab."""

        if self._dirty:
            self._emit_settings()
        super().hideEvent(event)

    def _update_tagger_inputs(self, name: str) -> None:
        is_wd14 = name.lower() == "wd14-onnx"
        self._tagger_model_edit.setEnabled(is_wd14)
        self._tagger_model_button.setEnabled(is_wd14)
        self._schedule_model_inspection()

    def set_pipeline(self, pipeline: ProcessingPipeline | None) -> None:
        self._pipeline = pipeline

    def set_tags_tab(self, tags_tab: TagsTab | None) -> None:
        self._tags_tab = tags_tab

    def _on_reset_database(self) -> None:
        db_path = self._view_model.db_path
        dialog = ResetDatabaseDialog(db_path, self)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return

        if self._tags_tab is not None and self._tags_tab.is_indexing_active():
            QMessageBox.warning(
                self,
                "Reset database",
                "Indexing is currently running. Please wait until it finishes before resetting.",
            )
            return

        backup = dialog.backup_enabled
        start_index = dialog.start_index_enabled

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to stop processing pipeline before reset")

        if self._tags_tab is not None:
            self._tags_tab.prepare_for_database_reset()

        try:
            result = self._view_model.reset_database(backup=backup)
        except Exception as exc:
            if self._tags_tab is not None:
                self._tags_tab.restore_connection()
            QMessageBox.critical(
                self,
                "Reset failed",
                (f"Database reset failed. Ensure no other process is accessing the database.\nDetails: {exc}"),
            )
            return

        if self._tags_tab is not None:
            self._tags_tab.handle_database_reset()

        backup_paths = [Path(path) for path in result.get("backup_paths", [])]
        message_lines = ["Database reset completed successfully."]
        if backup_paths:
            message_lines.append("Backups saved to:")
            message_lines.extend(f"  • {path}" for path in backup_paths)
        else:
            message_lines.append("No backup files were created.")

        QMessageBox.information(self, "Reset database", "\n".join(message_lines))

        if start_index and self._tags_tab is not None:
            self._tags_tab.start_indexing_now()

    def _on_browse_model(self) -> None:
        text_value = self._tagger_model_edit.text().strip()
        start_dir = str(Path(text_value).expanduser().parent) if text_value else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WD14 ONNX model",
            start_dir,
            "ONNX model (*.onnx);;All files (*)",
        )
        if file_path:
            self._tagger_model_edit.setText(file_path)

    def _schedule_model_inspection(self) -> None:
        self._inspection_generation += 1
        self._model_info_edit.setPlainText("Inspecting model configuration...")
        self._model_info_edit.setStyleSheet("")
        self._inspection_timer.start()

    def _start_model_inspection(self) -> None:
        if self._inspection_task is not None:
            self._inspection_task.cancel()
            self._inspection_task = None

        generation = self._inspection_generation
        tagger_name = self._tagger_combo.currentText()
        current = self._current_settings or PipelineSettings()
        tags_csv = current.tagger.tags_csv if tagger_name.lower() == "wd14-onnx" else None
        model_path = self._tagger_model_edit.text().strip() or None

        def _inspect(is_cancelled, _emit_progress) -> tuple[int, str, bool]:
            if is_cancelled():
                return generation, "", False
            try:
                inspection = inspect_model(
                    tagger_name=tagger_name,
                    model_path=model_path,
                    tags_csv=tags_csv,
                    provider_loader=self._view_model.provider_loader,
                )
                return generation, format_inspection(inspection), inspection.ok
            except Exception as exc:  # pragma: no cover - defensive UI fallback
                logger.exception("Failed to inspect tagger model")
                return generation, f"Model status: Error\nError: {exc}", False

        handle = self._inspection_jobs.submit_handle(
            CallableJob(_inspect, name="model-inspection"),
            priority=JobPriority.BACKGROUND,
        )
        self._inspection_task = handle

        def _clear_current() -> None:
            if self._inspection_task is handle:
                self._inspection_task = None

        def _done(payload) -> None:
            _clear_current()
            self._on_model_inspection_finished(*payload)

        handle.signals.completed.connect(_done)
        handle.signals.cancelled.connect(_clear_current)
        handle.signals.error.connect(lambda *_args: _clear_current())

    def _on_model_inspection_finished(self, generation: int, message: str, ok: bool) -> None:
        if generation != self._inspection_generation:
            return
        self._model_info_edit.setPlainText(message)
        if ok:
            self._model_info_edit.setStyleSheet("QPlainTextEdit { border: 1px solid #2e7d32; }")
        else:
            self._model_info_edit.setStyleSheet("QPlainTextEdit { border: 1px solid #c62828; }")


__all__ = ["SettingsTab"]
