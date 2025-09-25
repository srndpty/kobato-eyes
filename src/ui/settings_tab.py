"""Prototype settings UI for kobato-eyes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from core.config import load_settings, save_settings
from core.settings import EmbedModel, PipelineSettings, TaggerSettings
from db.admin import reset_database
from utils.paths import get_db_path

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

        self._backup_check = QCheckBox(
            "Backup .db / -wal / -shm before deleting (recommended)", self
        )
        self._backup_check.setChecked(True)

        self._purge_check = QCheckBox(
            "Delete HNSW index file (hnsw_cosine.bin) as well", self
        )
        self._purge_check.setChecked(True)

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
        layout.addWidget(self._purge_check)
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
    def purge_hnsw_enabled(self) -> bool:
        return self._purge_check.isChecked()

    @property
    def start_index_enabled(self) -> bool:
        return self._rescan_check.isChecked()


class SettingsTab(QWidget):
    """Provide minimal controls for pipeline configuration."""

    settings_applied = pyqtSignal(PipelineSettings)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._roots_edit = QPlainTextEdit(self)
        self._roots_edit.setPlaceholderText("One path per line")
        self._excluded_edit = QPlainTextEdit(self)
        self._excluded_edit.setPlaceholderText("Paths to ignore")

        self._hamming_spin = QSpinBox(self)
        self._hamming_spin.setRange(0, 64)
        self._hamming_spin.setValue(8)

        self._cosine_spin = QDoubleSpinBox(self)
        self._cosine_spin.setRange(0.0, 1.0)
        self._cosine_spin.setSingleStep(0.01)
        self._cosine_spin.setValue(0.2)

        self._ssim_spin = QDoubleSpinBox(self)
        self._ssim_spin.setRange(0.0, 1.0)
        self._ssim_spin.setSingleStep(0.01)
        self._ssim_spin.setValue(0.9)

        self._model_combo = QComboBox(self)
        self._model_combo.addItems(["ViT-L-14", "ViT-H-14", "RN50"])

        self._device_combo = QComboBox(self)
        self._device_combo.addItem("Auto", "auto")
        self._device_combo.addItem("CUDA", "cuda")
        self._device_combo.addItem("CPU", "cpu")

        self._pretrained_edit = QLineEdit(self)
        self._pretrained_edit.setPlaceholderText("e.g. openai")

        self._auto_index_check = QCheckBox("Auto index changes", self)
        self._auto_index_check.setChecked(True)

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
        form.addRow("Hamming threshold", self._hamming_spin)
        form.addRow("Cosine threshold", self._cosine_spin)
        form.addRow("SSIM threshold", self._ssim_spin)
        form.addRow("Model", self._model_combo)
        form.addRow("Device", self._device_combo)
        form.addRow("Pretrained tag", self._pretrained_edit)
        form.addRow("Tagger", self._tagger_combo)
        form.addRow("Model path", tagger_model_row)
        form.addRow("", tagger_env_row)
        form.addRow(self._auto_index_check)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch()
        buttons = QHBoxLayout()
        buttons.addWidget(self._reset_button)
        buttons.addStretch()
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)

        self._current_settings: PipelineSettings = PipelineSettings()
        self._pipeline: ProcessingPipeline | None = None
        self._tags_tab: TagsTab | None = None
        self._load_initial_settings()

    def load_settings(self, settings: PipelineSettings) -> None:
        self._current_settings = settings
        self._roots_edit.setPlainText("\n".join(str(path) for path in settings.roots))
        self._excluded_edit.setPlainText("\n".join(str(path) for path in settings.excluded))
        self._hamming_spin.setValue(settings.hamming_threshold)
        self._cosine_spin.setValue(settings.cosine_threshold)
        self._ssim_spin.setValue(settings.ssim_threshold)
        self._auto_index_check.setChecked(bool(settings.auto_index))
        index = self._model_combo.findText(settings.model_name)
        if index >= 0:
            self._model_combo.setCurrentIndex(index)
        self._pretrained_edit.setText(settings.embed_model.pretrained)
        device_index = self._device_combo.findData(settings.embed_model.device)
        if device_index >= 0:
            self._device_combo.setCurrentIndex(device_index)
        else:
            self._device_combo.setCurrentIndex(0)
        tagger_index = self._tagger_combo.findText(settings.tagger.name)
        if tagger_index >= 0:
            self._tagger_combo.setCurrentIndex(tagger_index)
        else:
            self._tagger_combo.setCurrentText(settings.tagger.name)
        self._tagger_model_edit.setText(settings.tagger.model_path or "")
        self._update_tagger_inputs(self._tagger_combo.currentText())

    def _emit_settings(self) -> None:
        current = self._current_settings
        previous_tagger = current.tagger if current else TaggerSettings()
        tagger_name = self._tagger_combo.currentText()
        is_wd14 = tagger_name.lower() == "wd14-onnx"
        model_path_text = self._tagger_model_edit.text().strip()
        model_path = model_path_text if is_wd14 and model_path_text else None
        tagger_settings = TaggerSettings(
            name=tagger_name,
            model_path=model_path,
            tags_csv=previous_tagger.tags_csv if is_wd14 else None,
            thresholds=dict(previous_tagger.thresholds),
        )
        settings = PipelineSettings(
            roots=[Path(line) for line in self._lines(self._roots_edit) if line],
            excluded=[Path(line) for line in self._lines(self._excluded_edit) if line],
            hamming_threshold=int(self._hamming_spin.value()),
            cosine_threshold=float(self._cosine_spin.value()),
            ssim_threshold=float(self._ssim_spin.value()),
            embed_model=EmbedModel(
                name=self._model_combo.currentText(),
                pretrained=self._pretrained_edit.text().strip(),
                device=str(self._device_combo.currentData()),
            ),
            auto_index=self._auto_index_check.isChecked(),
            tagger=tagger_settings,
        )
        save_settings(settings)
        self._current_settings = settings
        self.settings_applied.emit(settings)

    def _load_initial_settings(self) -> None:
        settings = load_settings()
        self.load_settings(settings)

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
        db_path = get_db_path()
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
        purge = dialog.purge_hnsw_enabled
        start_index = dialog.start_index_enabled

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to stop processing pipeline before reset")

        if self._tags_tab is not None:
            self._tags_tab.prepare_for_database_reset()

        try:
            result = reset_database(db_path, backup=backup, purge_hnsw=purge)
        except Exception as exc:
            logger.exception("Database reset failed for %s", db_path)
            if self._tags_tab is not None:
                self._tags_tab.restore_connection()
            QMessageBox.critical(
                self,
                "Reset failed",
                (
                    "Database reset failed. Ensure no other process is accessing the database.\n"
                    f"Details: {exc}"
                ),
            )
            return

        if self._tags_tab is not None:
            self._tags_tab.handle_database_reset()

        backup_paths = [Path(path) for path in result.get("backup_paths", [])]
        hnsw_deleted = bool(result.get("hnsw_deleted", False))
        message_lines = ["Database reset completed successfully."]
        if backup_paths:
            message_lines.append("Backups saved to:")
            message_lines.extend(f"  • {path}" for path in backup_paths)
        else:
            message_lines.append("No backup files were created.")
        if purge:
            if hnsw_deleted:
                message_lines.append("HNSW index file was deleted.")
            else:
                message_lines.append("HNSW index file was not found.")

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
        try:
            from tagger import wd14_onnx

            providers = wd14_onnx.get_available_providers()
        except RuntimeError as exc:
            message = str(exc)
            logger.warning("Tagger environment check failed: %s", exc)
        else:
            if providers:
                joined = ", ".join(providers)
            else:
                joined = "<none>"
            message = f"ONNX providers: {joined}"
            logger.info("Available ONNX providers: %s", joined)
        self._show_environment_message(message)

    def _show_environment_message(self, message: str) -> None:
        rect = self._tagger_env_button.rect()
        global_pos = self._tagger_env_button.mapToGlobal(rect.center())
        QToolTip.showText(global_pos, message, self._tagger_env_button, rect, 4000)


__all__ = ["SettingsTab"]
