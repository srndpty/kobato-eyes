"""Prototype settings UI for kobato-eyes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.pipeline import PipelineSettings


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
        self._model_combo.addItems(
            [
                "clip-vit",
                "ViT-H-14",
                "RN50",
            ]
        )

        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self._emit_settings)

        form = QFormLayout()
        form.addRow(QLabel("Roots"), self._roots_edit)
        form.addRow(QLabel("Excluded"), self._excluded_edit)
        form.addRow("Hamming threshold", self._hamming_spin)
        form.addRow("Cosine threshold", self._cosine_spin)
        form.addRow("SSIM threshold", self._ssim_spin)
        form.addRow("Model", self._model_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch()
        layout.addWidget(apply_button)

    def load_settings(self, settings: PipelineSettings) -> None:
        self._roots_edit.setPlainText("\n".join(str(path) for path in settings.roots))
        self._excluded_edit.setPlainText("\n".join(str(path) for path in settings.excluded))
        self._hamming_spin.setValue(settings.hamming_threshold)
        self._cosine_spin.setValue(settings.cosine_threshold)
        self._ssim_spin.setValue(settings.ssim_threshold)
        index = self._model_combo.findText(settings.model_name)
        if index >= 0:
            self._model_combo.setCurrentIndex(index)

    def _emit_settings(self) -> None:
        settings = PipelineSettings(
            roots=[Path(line) for line in self._lines(self._roots_edit) if line],
            excluded=[Path(line) for line in self._lines(self._excluded_edit) if line],
            hamming_threshold=self._hamming_spin.value(),
            cosine_threshold=self._cosine_spin.value(),
            ssim_threshold=self._ssim_spin.value(),
            model_name=self._model_combo.currentText(),
        )
        self.settings_applied.emit(settings)

    @staticmethod
    def _lines(edit: QPlainTextEdit) -> Iterable[str]:
        return (line.strip() for line in edit.toPlainText().splitlines())


__all__ = ["SettingsTab"]
