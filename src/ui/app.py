"""Minimal PyQt6 application entry point for kobato-eyes."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from utils.env import is_headless
from utils.paths import ensure_dirs, get_app_paths, migrate_data_dir_if_needed

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
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication
    from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

    QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    from core.config import PipelineSettings
    from db.connection import bootstrap_if_needed, get_conn
    from ui.dup_tab import DupTab
    from ui.settings_tab import SettingsTab
    from ui.tags_tab import TagsTab

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

    def _quick_settle_sqlite(db_path: str | os.PathLike) -> None:
        try:
            conn = get_conn(db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=2000")
                # まず PASSIVE で消化
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                # アイドルなら一気に 0 に（ダメでも害なし）
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                # FTS があれば軽く最適化（速い）
                conn.execute("PRAGMA optimize")
            finally:
                conn.close()
        except Exception:
            # 起動ブロックは避けたいのでログだけで継続
            import logging

            logging.getLogger(__name__).exception("quick_settle_sqlite failed")

    class MainWindow(QMainWindow):
        """Main window presenting basic navigation tabs."""

        def __init__(self) -> None:
            super().__init__()
            migrate_data_dir_if_needed()
            ensure_dirs()
            app_paths = get_app_paths()
            db_path = app_paths.db_path()
            logger.info("DB at %s", db_path)
            _quick_settle_sqlite(db_path)
            bootstrap_if_needed(db_path)
            self.setWindowTitle("kobato-eyes")
            self._tabs = QTabWidget()
            self._tags_tab = TagsTab(self)
            self._dup_tab = DupTab(self)
            self._settings_tab = SettingsTab(self)
            self._settings_tab.settings_applied.connect(self._on_settings_applied)
            self._settings_tab.set_tags_tab(self._tags_tab)
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
        _install_crash_handlers()

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
