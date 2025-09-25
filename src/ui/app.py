"""Minimal PyQt6 application entry point for kobato-eyes."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from utils.env import is_headless
from utils.paths import ensure_dirs, get_db_path, get_log_dir, migrate_data_dir_if_needed

HEADLESS = is_headless()

if HEADLESS:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_OPENGL", "software")

logger = logging.getLogger(__name__)


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

    log_dir = get_log_dir()
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
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication
    from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

    QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    from db.connection import bootstrap_if_needed
    from core.settings import PipelineSettings
    from ui.dup_tab import DupTab
    from ui.settings_tab import SettingsTab
    from ui.tags_tab import TagsTab

    class MainWindow(QMainWindow):
        """Main window presenting basic navigation tabs."""

        def __init__(self) -> None:
            super().__init__()
            migrate_data_dir_if_needed()
            ensure_dirs()
            db_path = get_db_path()
            logger.info("DB at %s", db_path)
            bootstrap_if_needed(db_path)
            self.setWindowTitle("kobato-eyes")
            self._tabs = QTabWidget()
            self._tags_tab = TagsTab(self)
            self._dup_tab = DupTab(self)
            self._settings_tab = SettingsTab(self)
            self._settings_tab.settings_applied.connect(self._on_settings_applied)
            self._tabs.addTab(self._tags_tab, "Tags")
            self._tabs.addTab(self._dup_tab, "Duplicates")
            self._tabs.addTab(self._settings_tab, "Settings")
            self.setCentralWidget(self._tabs)
            self._init_menus()

        def _init_menus(self) -> None:
            help_menu = self.menuBar().addMenu("Help")
            open_logs_action = QAction("Open logs folder", self)
            open_logs_action.triggered.connect(self._open_logs_folder)
            help_menu.addAction(open_logs_action)

        def _open_logs_folder(self) -> None:
            log_dir = get_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

        def _on_settings_applied(self, settings: PipelineSettings) -> None:
            self._tags_tab.reload_autocomplete(settings)


    def run() -> None:
        """Launch the kobato-eyes GUI application."""

        setup_logging()
        app = QApplication(sys.argv)

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
        window = MainWindow()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    run()
