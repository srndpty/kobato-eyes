"""Animated splash screen used during application startup."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QPixmap, QTransform
from PyQt6.QtWidgets import QLabel, QWidget

_SPLASH_IMAGE: Final[Path] = Path(__file__).resolve().parent / "assets" / "splash" / "splash.png"


class AnimatedSplashScreen(QWidget):
    """Simple splash screen that alternates between two rotations of an image."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(512, 512)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setFixedSize(512, 512)

        self._pixmaps = self._load_pixmaps()
        self._current_index = 0
        self._label.setPixmap(self._pixmaps[self._current_index])

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._advance_frame)

    def event(self, event: QEvent) -> bool:  # noqa: D401 - Qt override doc inherited
        if event.type() == QEvent.Type.Show:
            self._centre_on_primary_screen()
            self._timer.start()
        elif event.type() == QEvent.Type.Hide:
            self._timer.stop()
        return super().event(event)

    def finish(self, parent: QWidget | None) -> None:
        """Stop the animation and close the splash screen."""

        if self._timer.isActive():
            self._timer.stop()
        self.close()
        if parent is not None:
            parent.raise_()
            parent.activateWindow()

    def _centre_on_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geometry = screen.availableGeometry()
        x = geometry.x() + (geometry.width() - self.width()) // 2
        y = geometry.y() + (geometry.height() - self.height()) // 2
        self.move(QPoint(x, y))

    def _advance_frame(self) -> None:
        self._current_index = (self._current_index + 1) % len(self._pixmaps)
        self._label.setPixmap(self._pixmaps[self._current_index])

    def _load_pixmaps(self) -> list[QPixmap]:
        base = QPixmap(str(_SPLASH_IMAGE))
        if base.isNull():
            # fallback to empty pixmap to avoid crash
            base = QPixmap(512, 512)
            base.fill(Qt.GlobalColor.transparent)
        base = base.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        rotations = (-20, -40)
        pixmaps: list[QPixmap] = []
        for angle in rotations:
            transform = QTransform().rotate(angle)
            rotated = base.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            pixmaps.append(rotated.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        return pixmaps
