"""Animated splash screen displayed during application startup."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PyQt6.QtCore import QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QGuiApplication, QPainter, QPixmap, QShowEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class RotatingSplashScreen(QWidget):
    """Splash screen that alternates the rotation of the application logo."""

    _IMAGE_SIZE: Final[int] = 512
    _ANGLES: Final[tuple[float, float]] = (-20.0, -40.0)

    def __init__(self, image_path: str | Path | None = None) -> None:
        super().__init__(
            None,
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName("kobato_eyes_splash")

        resolved_path = self._resolve_image_path(image_path)
        self._base_pixmap = self._load_base_pixmap(resolved_path)
        self._pixmaps = tuple(
            self._create_rotated_pixmap(angle) for angle in self._ANGLES
        )
        self._current_index = 0

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self.setFixedSize(self._IMAGE_SIZE, self._IMAGE_SIZE)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._advance_frame)
        self._apply_pixmap()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        """Start the animation and center the splash on the primary screen."""

        self._center_on_screen()
        if len(self._pixmaps) > 1 and not self._timer.isActive():
            self._timer.start()
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Stop the animation when the splash is closed."""

        if self._timer.isActive():
            self._timer.stop()
        super().closeEvent(event)

    def finish(self, widget: QWidget | None = None) -> None:
        """Stop the animation and close the splash screen."""

        if self._timer.isActive():
            self._timer.stop()
        self.close()
        if widget is not None:
            widget.activateWindow()
            widget.raise_()

    def _advance_frame(self) -> None:
        self._current_index = (self._current_index + 1) % len(self._pixmaps)
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        self._label.setPixmap(self._pixmaps[self._current_index])

    def _resolve_image_path(self, image_path: str | Path | None) -> Path:
        if image_path is not None:
            return Path(image_path)
        return Path(__file__).resolve().parent / "assets" / "splash" / "splash.png"

    def _load_base_pixmap(self, image_path: Path) -> QPixmap:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            placeholder = QPixmap(self._IMAGE_SIZE, self._IMAGE_SIZE)
            placeholder.fill(Qt.GlobalColor.transparent)
            return placeholder
        return pixmap.scaled(
            self._IMAGE_SIZE,
            self._IMAGE_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _create_rotated_pixmap(self, angle: float) -> QPixmap:
        target = QPixmap(self._IMAGE_SIZE, self._IMAGE_SIZE)
        target.fill(Qt.GlobalColor.transparent)

        painter = QPainter(target)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        center = QPoint(self._IMAGE_SIZE // 2, self._IMAGE_SIZE // 2)
        painter.translate(center)
        painter.rotate(angle)
        half_width = self._base_pixmap.width() / 2
        half_height = self._base_pixmap.height() / 2
        painter.drawPixmap(QPointF(-half_width, -half_height), self._base_pixmap)
        painter.end()
        return target

    def _center_on_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        x = geometry.x() + (geometry.width() - self.width()) // 2
        y = geometry.y() + (geometry.height() - self.height()) // 2
        self.move(x, y)


__all__ = ["RotatingSplashScreen"]

