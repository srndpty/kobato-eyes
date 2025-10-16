"""Minimal PyQt6 application entry point for kobato-eyes."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.env import is_headless
from utils.paths import get_app_paths

HEADLESS = is_headless()

if HEADLESS:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_OPENGL", "software")

logger = logging.getLogger(__name__)
# import faulthandler, threading
# faulthandler.enable()
# threading.Timer(60, faulthandler.dump_traceback).start()


def _resolve_log_level(value: str | None) -> int:
    if not value:
        return logging.INFO
    level = logging.getLevelName(value.upper())
    if isinstance(level, int):
        return level
    return logging.INFO


def setup_logging() -> None:
    """Configure logging to stdout and a rotating application log file."""

    level = _resolve_log_level(os.environ.get("KOE_LOG_LEVEL"))
    root_logger = logging.getLogger()

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    app_paths = get_app_paths()
    log_dir = app_paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


if HEADLESS:

    class MainWindow:  # type: ignore[too-many-ancestors]
        """Placeholder window used when Qt is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - Qt-compatible signature
            raise RuntimeError("kobato-eyes UI is unavailable in headless mode")

    def run() -> None:
        """Headless environments cannot launch the GUI."""

        raise RuntimeError("kobato-eyes UI is unavailable in headless mode")


else:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap, QTransform
    from PyQt6.QtWidgets import QApplication, QMainWindow, QSplashScreen, QTabWidget, QWidget

    QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    from ui.dup_tab import DupTab
    from ui.icons import EyeIconProvider
    from ui.settings_tab import SettingsTab
    from ui.tags_tab import TagsTab
    from ui.viewmodels import MainViewModel

    def _install_crash_handlers():
        import atexit
        import faulthandler
        import os
        import signal
        import sys
        import threading
        import time
        import traceback
        from pathlib import Path

        log_dir = Path(os.environ.get("APPDATA", ".")) / "kobato-eyes" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "crash.log"

        # すべてのスレッドのスタックを致命的シグナルで吐く
        f = open(log_file, "a", encoding="utf-8", buffering=1)
        try:
            faulthandler.enable(all_threads=True, file=f)
        except Exception:
            pass

        def _dump_with_header(header: str):
            try:
                f.write(f"\n==== {header} {time.ctime()} ====\n")
                faulthandler.dump_traceback(file=f, all_threads=True)
                f.flush()
            except Exception:
                pass

        def _sys_excepthook(t, v, tb):
            try:
                f.write(f"\n==== sys.excepthook {time.ctime()} ====\n")
                traceback.print_exception(t, v, tb, file=f)
                f.flush()
            except Exception:
                pass

        sys.excepthook = _sys_excepthook

        def _threading_excepthook(args):
            _sys_excepthook(args.exc_type, args.exc_value, args.exc_traceback)

        try:
            threading.excepthook = _threading_excepthook
        except Exception:
            pass

        for sname in ("SIGABRT", "SIGTERM", "SIGBREAK"):
            s = getattr(signal, sname, None)
            if s:
                try:
                    signal.signal(s, lambda signum, frame: _dump_with_header(f"signal {signum}"))
                except Exception:
                    pass

        atexit.register(lambda: f.write(f"\n==== process exit {time.ctime()} ====\n") or f.flush())

    class AnimatedSplashScreen(QSplashScreen):
        """Splash screen that alternates between two rotated frames."""

        _FRAME_INTERVAL_MS = 500
        _FRAME_SIZE = 512
        _ROTATION_ANGLES = (-20, -40)

        def __init__(self) -> None:
            pixmap_path = Path(__file__).resolve().parent / "assets" / "splash" / "splash.png"
            original = QPixmap(str(pixmap_path))
            if original.isNull():
                raise FileNotFoundError(f"Splash image not found: {pixmap_path}")

            frames = [self._create_frame(original, angle) for angle in self._ROTATION_ANGLES]
            super().__init__(frames[0])

            self._frames = frames
            self._frame_index = 0
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setFixedSize(self.pixmap().size())
            self._timer = QTimer(self)
            self._timer.setInterval(self._FRAME_INTERVAL_MS)
            self._timer.timeout.connect(self._advance_frame)
            self._timer.start()
            self._center_on_primary_screen()

        def _create_frame(self, original: QPixmap, angle: float) -> QPixmap:
            transform = QTransform()
            transform.rotate(angle)
            rotated = original.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            return rotated.scaled(
                self._FRAME_SIZE,
                self._FRAME_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        def _advance_frame(self) -> None:
            self._frame_index = (self._frame_index + 1) % len(self._frames)
            self.setPixmap(self._frames[self._frame_index])
            self.setFixedSize(self.pixmap().size())
            self._center_on_primary_screen()

        def _center_on_primary_screen(self) -> None:
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                return
            geometry = screen.availableGeometry()
            x = geometry.x() + (geometry.width() - self.width()) // 2
            y = geometry.y() + (geometry.height() - self.height()) // 2
            self.move(x, y)

        def finish(self, widget: QWidget | None = None) -> None:
            self._timer.stop()
            super().finish(widget)

    class MainWindow(QMainWindow):
        """Main window presenting basic navigation tabs."""

        def __init__(
            self,
            view_model: MainViewModel | None = None,
            icon_provider: EyeIconProvider | None = None,
        ) -> None:
            super().__init__()
            self._view_model = view_model or MainViewModel(self)
            db_path = self._view_model.db_path
            logger.info("DB at %s", db_path)
            self._tags_view_model = self._view_model.create_tags_view_model(self)
            self._dup_view_model = self._view_model.create_dup_view_model(self)
            self._settings_view_model = self._view_model.create_settings_view_model(self)
            self.setWindowTitle("kobato-eyes")
            self._icon_provider: EyeIconProvider = icon_provider or EyeIconProvider()
            self._tabs = QTabWidget()
            self._tags_tab = TagsTab(self, view_model=self._tags_view_model)
            self._dup_tab = DupTab(self, view_model=self._dup_view_model)
            self._settings_tab = SettingsTab(self, view_model=self._settings_view_model)
            self._view_model.settings_changed.connect(self._tags_tab.reload_autocomplete)
            self._settings_tab.set_tags_tab(self._tags_tab)
            self._tabs.addTab(self._tags_tab, "Tags")
            self._tabs.addTab(self._dup_tab, "Duplicates")
            self._tabs.addTab(self._settings_tab, "Settings")
            self.setCentralWidget(self._tabs)
            self._dup_tab_index: int = self._tabs.indexOf(self._dup_tab)
            self._tabs.currentChanged.connect(self._handle_tab_change)
            self._handle_tab_change(self._tabs.currentIndex())

        def _handle_tab_change(self, index: int) -> None:
            """Update the window and taskbar icon based on the active tab."""

            if index == self._dup_tab_index:
                icon = self._icon_provider.left_eye
            else:
                icon = self._icon_provider.right_eye
            self._apply_window_icon(icon)

        def _apply_window_icon(self, icon: QIcon) -> None:
            """Apply the given icon to the window and the running QApplication."""

            self.setWindowIcon(icon)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)

    def run() -> None:
        """Launch the kobato-eyes GUI application."""

        setup_logging()
        _install_crash_handlers()

        app = QApplication(sys.argv)
        icon_provider = EyeIconProvider()
        app.setWindowIcon(icon_provider.right_eye)

        splash = AnimatedSplashScreen()
        splash.show()
        app.processEvents()

        def _finalise_onnx_profiles() -> None:
            try:
                from tagger.wd14_onnx import end_all_profiles
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to import wd14 profiler finaliser")
                return
            try:
                end_all_profiles()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to flush WD14 ONNX profiles")

        app.aboutToQuit.connect(_finalise_onnx_profiles)
        window = MainWindow(icon_provider=icon_provider)
        window.resize(1024, 768)
        window.show()
        splash.finish(window)
        sys.exit(app.exec())


if __name__ == "__main__":
    run()
