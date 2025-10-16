"""Animated splash screen used during application startup."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QGuiApplication, QPainter, QPixmap, QShowEvent
from PyQt6.QtWidgets import QSplashScreen, QWidget

_SPLASH_DIR: Final[Path] = Path(__file__).resolve().parent / "assets" / "splash"
_SPLASH_FILE_NAME: Final[str] = "splash.png"
_SPLASH_SIZE: Final[int] = 512
_ANIMATION_INTERVAL_MS: Final[int] = 500
_DEFAULT_ANGLES: Final[tuple[float, ...]] = (-20.0, -40.0)


def resolve_default_splash_path() -> Path:
    """Return the absolute path to the bundled splash image."""

    path = _SPLASH_DIR / _SPLASH_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(path)
    return path


class AnimatedSplashScreen(QSplashScreen):
    """Splash screen that alternates rotated frames of a single pixmap."""

    def __init__(
        self,
        pixmap_path: Path,
        *,
        angles: Iterable[float] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self._base_pixmap = self._load_base_pixmap(pixmap_path)
        self._angles = tuple(angles or _DEFAULT_ANGLES)
        if not self._angles:
            raise ValueError("At least one angle must be provided for the splash animation")
        self._frames = tuple(self._create_frame(angle) for angle in self._angles)
        if not self._frames:
            raise RuntimeError("Failed to build splash animation frames")
        flags = (
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(parent, self._frames[0], flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._current_frame = 0
        self._timer.start(_ANIMATION_INTERVAL_MS)

    def showEvent(self, event: QShowEvent) -> None:
        """Centre the splash on the primary screen when it is shown."""

        super().showEvent(event)
        self._centre_on_screen()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the animation timer when the splash closes."""

        if self._timer.isActive():
            self._timer.stop()
        super().closeEvent(event)

    def finish(self, widget: QWidget | None) -> None:  # type: ignore[override]
        """Stop the animation before delegating to :class:`QSplashScreen`."""

        if self._timer.isActive():
            self._timer.stop()
        super().finish(widget)

    def _load_base_pixmap(self, pixmap_path: Path) -> QPixmap:
        pixmap = QPixmap(str(pixmap_path))
        if pixmap.isNull():
            raise FileNotFoundError(pixmap_path)
        return pixmap.scaled(
            _SPLASH_SIZE,
            _SPLASH_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _create_frame(self, angle: float) -> QPixmap:
        frame = QPixmap(_SPLASH_SIZE, _SPLASH_SIZE)
        frame.fill(Qt.GlobalColor.transparent)
        painter = QPainter(frame)
        try:
            painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.SmoothPixmapTransform
                | QPainter.RenderHint.HighQualityAntialiasing,
                True,
            )
            painter.translate(_SPLASH_SIZE / 2, _SPLASH_SIZE / 2)
            painter.rotate(angle)
            painter.translate(-self._base_pixmap.width() / 2, -self._base_pixmap.height() / 2)
            painter.drawPixmap(0, 0, self._base_pixmap)
        finally:
            painter.end()
        return frame

    def _advance_frame(self) -> None:
        if not self._frames:
            return
        self._current_frame = (self._current_frame + 1) % len(self._frames)
        self.setPixmap(self._frames[self._current_frame])

    def _centre_on_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        x = geometry.x() + (geometry.width() - self.width()) // 2
        y = geometry.y() + (geometry.height() - self.height()) // 2
        self.move(x, y)


__all__ = ["AnimatedSplashScreen", "resolve_default_splash_path"]
