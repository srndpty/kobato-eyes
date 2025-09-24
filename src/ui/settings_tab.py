"""Prototype settings UI for kobato-eyes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import load_settings, save_settings
from core.settings import EmbedModel, PipelineSettings, TaggerSettings


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

        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self._emit_settings)

        form = QFormLayout()
        form.addRow(QLabel("Roots"), self._roots_edit)
        form.addRow(QLabel("Excluded"), self._excluded_edit)
        form.addRow("Hamming threshold", self._hamming_spin)
        form.addRow("Cosine threshold", self._cosine_spin)
        form.addRow("SSIM threshold", self._ssim_spin)
        form.addRow("Model", self._model_combo)
        form.addRow("Pretrained tag", self._pretrained_edit)
        form.addRow("Tagger", self._tagger_combo)
        form.addRow("Model path", tagger_model_row)
        form.addRow(self._auto_index_check)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch()
        layout.addWidget(apply_button)

        self._current_settings: PipelineSettings = PipelineSettings()
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


__all__ = ["SettingsTab"]
