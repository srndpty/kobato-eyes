"""Reusable semi-transparent overlay with progress feedback."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QRect, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QLabel, QProgressBar, QVBoxLayout, QWidget

DEFAULT_OVERLAY_MESSAGE = "Searching… (Esc to cancel)"


def overlay_message(message: str | None) -> str:
    """Return the visible overlay message."""

    return message if message is not None else DEFAULT_OVERLAY_MESSAGE


def overlay_geometry(parent: QWidget | None) -> QRect:
    """Return the geometry the overlay should occupy."""

    if parent is None:
        return QRect()
    return parent.rect()


class SpinnerOverlay(QWidget):
    """Display an indeterminate progress indicator above a widget."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAutoFillBackground(False)

        self._message_label = QLabel(DEFAULT_OVERLAY_MESSAGE, self)
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setStyleSheet("color: white; font-size: 14px; font-weight: 600;")
        self._progress = QProgressBar(self)
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)

        self._bg_color = QColor(0, 0, 0, 128)

        # 中央のパネル
        panel = QFrame(self)
        panel.setObjectName("overlayPanel")
        panel.setStyleSheet("#overlayPanel { background: #2b2b2b; border-radius: 12px; }")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.addWidget(self._message_label, 0, Qt.AlignmentFlag.AlignCenter)
        panel_layout.addSpacing(12)
        panel_layout.addWidget(self._progress, 0, Qt.AlignmentFlag.AlignCenter)

        shadow = QGraphicsDropShadowEffect(panel)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 160))
        panel.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.addStretch(1)
        layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        self.hide()
        parent.installEventFilter(self)

    def paintEvent(self, event) -> None:
        """全面を不透明で塗る。スタイルや属性に依存しないので確実。"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color)
        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition()

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self.setGeometry(overlay_geometry(self.parentWidget()))
        return super().eventFilter(obj, event)

    def show(self, message: str | None = None) -> None:  # type: ignore[override]
        self._message_label.setText(overlay_message(message))
        self._progress.setRange(0, 0)
        self._progress.setValue(0)
        self._reposition()
        super().show()
        self.raise_()

    def set_message(self, message: str) -> None:
        self._message_label.setText(message)

    def hide(self) -> None:  # type: ignore[override]
        super().hide()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        self.setGeometry(overlay_geometry(parent))


__all__ = ["DEFAULT_OVERLAY_MESSAGE", "SpinnerOverlay", "overlay_geometry", "overlay_message"]
