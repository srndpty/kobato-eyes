"""Reusable semi-transparent overlay with progress feedback."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class SpinnerOverlay(QWidget):
    """Display indeterminate or determinate progress above a widget."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 128);")
        self._message_label = QLabel("Searching…", self)
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setStyleSheet("color: white; font-weight: 500;")
        self._progress = QProgressBar(self)
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress.setTextVisible(True)
        self._progress.setRange(0, 0)
        self._progress.setFormat("%p%")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.addStretch(1)
        layout.addWidget(self._message_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(12)
        layout.addWidget(self._progress, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        self.hide()
        self._determinate = False
        parent.installEventFilter(self)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parentWidget().rect())
        return super().eventFilter(obj, event)

    def show_indeterminate(self, message: str) -> None:
        self._determinate = False
        self._message_label.setText(message)
        self._progress.setRange(0, 0)
        self._progress.setFormat("%p%")
        self._progress.setValue(0)
        self._reposition()
        self.show()
        self.raise_()

    def show_determinate(self, message: str, maximum: int) -> None:
        self._determinate = True
        self._message_label.setText(message)
        maximum = max(0, int(maximum))
        self._progress.setRange(0, maximum if maximum > 0 else 1)
        if maximum <= 0:
            self._progress.setValue(0)
        self._reposition()
        self.show()
        self.raise_()

    def update_value(self, value: int) -> None:
        if not self._determinate:
            return
        maximum = self._progress.maximum()
        clamped = max(0, min(int(value), maximum))
        self._progress.setValue(clamped)

    def hide_overlay(self) -> None:
        self.hide()
        self._determinate = False

    def is_determinate(self) -> bool:
        return self._determinate

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.setGeometry(parent.rect())


__all__ = ["SpinnerOverlay"]
