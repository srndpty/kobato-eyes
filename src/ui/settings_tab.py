"""Prototype settings UI for kobato-eyes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from PyQt6.QtCore import pyqtSignal
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
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from core.config import PipelineSettings, TaggerSettings
from ui.viewmodels import SettingsViewModel

if TYPE_CHECKING:
    from core.pipeline import ProcessingPipeline
    from ui.tags_tab import TagsTab


logger = logging.getLogger(__name__)


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
        self._roots_edit = QPlainTextEdit(self)
        self._roots_edit.setPlaceholderText("One path per line")
        self._excluded_edit = QPlainTextEdit(self)
        self._excluded_edit.setPlaceholderText("Paths to ignore")

        self._device_combo = QComboBox(self)
        self._device_combo.addItem("Auto", "auto")
        self._device_combo.addItem("CUDA", "cuda")
        self._device_combo.addItem("CPU", "cpu")

        self._tagger_combo = QComboBox(self)
        self._tagger_combo.addItems(["dummy", "wd14-onnx"])
        self._tagger_combo.currentTextChanged.connect(self._update_tagger_inputs)

        self._tagger_model_edit = QLineEdit(self)
        self._tagger_model_edit.setPlaceholderText("Path to WD14 ONNX model")
        self._tagger_model_button = QPushButton("Browse…", self)
        self._tagger_model_button.clicked.connect(self._on_browse_model)
        tagger_model_row = QWidget(self)
        tagger_layout = QHBoxLayout(tagger_model_row)
        tagger_layout.setContentsMargins(0, 0, 0, 0)
        tagger_layout.addWidget(self._tagger_model_edit)
        tagger_layout.addWidget(self._tagger_model_button)

        self._tagger_env_button = QPushButton("Check environment", self)
        self._tagger_env_button.setToolTip("Log and display available ONNX providers")
        self._tagger_env_button.clicked.connect(self._on_check_tagger_env)
        tagger_env_row = QWidget(self)
        tagger_env_layout = QHBoxLayout(tagger_env_row)
        tagger_env_layout.setContentsMargins(0, 0, 0, 0)
        tagger_env_layout.addWidget(self._tagger_env_button)
        tagger_env_layout.addStretch()

        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self._emit_settings)

        self._reset_button = QPushButton("Reset database…", self)
        self._reset_button.clicked.connect(self._on_reset_database)

        form = QFormLayout()
        form.addRow(QLabel("Roots"), self._roots_edit)
        form.addRow(QLabel("Excluded"), self._excluded_edit)
        form.addRow("Device", self._device_combo)
        form.addRow("Tagger", self._tagger_combo)
        form.addRow("Model path", tagger_model_row)
        form.addRow("", tagger_env_row)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch()
        buttons = QHBoxLayout()
        buttons.addWidget(self._reset_button)
        buttons.addStretch()
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)

        self._current_settings: PipelineSettings = self._view_model.current_settings
        self._pipeline: ProcessingPipeline | None = None
        self._tags_tab: TagsTab | None = None
        self.load_settings(self._current_settings)

    def load_settings(self, settings: PipelineSettings) -> None:
        self._current_settings = settings
        self._view_model.set_current_settings(settings)
        self._roots_edit.setPlainText("\n".join(str(path) for path in settings.roots))
        self._excluded_edit.setPlainText("\n".join(str(path) for path in settings.excluded))

        tagger_index = self._tagger_combo.findText(settings.tagger.name)
        if tagger_index >= 0:
            self._tagger_combo.setCurrentIndex(tagger_index)
        else:
            self._tagger_combo.setCurrentText(settings.tagger.name)
        self._tagger_model_edit.setText(settings.tagger.model_path or "")
        self._update_tagger_inputs(self._tagger_combo.currentText())

    def _emit_settings(self) -> None:
        current = self._current_settings or PipelineSettings()
        previous_tagger = current.tagger if current else TaggerSettings()
        tagger_name = self._tagger_combo.currentText()
        is_wd14 = tagger_name.lower() == "wd14-onnx"
        model_path_text = self._tagger_model_edit.text().strip()
        model_path = model_path_text if is_wd14 and model_path_text else None
        settings = self._view_model.build_settings(
            roots=[Path(line) for line in self._lines(self._roots_edit) if line],
            excluded=[Path(line) for line in self._lines(self._excluded_edit) if line],
            tagger_name=tagger_name,
            model_path=model_path,
            previous_tagger=previous_tagger,
        )
        self._current_settings = settings
        self._view_model.apply_settings(settings)

    @staticmethod
    def _lines(edit: QPlainTextEdit) -> Iterable[str]:
        return (line.strip() for line in edit.toPlainText().splitlines())

    def _update_tagger_inputs(self, name: str) -> None:
        is_wd14 = name.lower() == "wd14-onnx"
        self._tagger_model_edit.setEnabled(is_wd14)
        self._tagger_model_button.setEnabled(is_wd14)
        self._tagger_env_button.setEnabled(is_wd14)

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
                ("Database reset failed. Ensure no other process is accessing the database.\n" f"Details: {exc}"),
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

    def _on_check_tagger_env(self) -> None:
        message = self._view_model.check_tagger_environment()
        self._show_environment_message(message)

    def _show_environment_message(self, message: str) -> None:
        rect = self._tagger_env_button.rect()
        global_pos = self._tagger_env_button.mapToGlobal(rect.center())
        QToolTip.showText(global_pos, message, self._tagger_env_button, rect, 4000)


__all__ = ["SettingsTab"]
