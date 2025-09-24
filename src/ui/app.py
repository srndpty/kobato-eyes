"""Minimal PyQt6 application entry point for kobato-eyes."""

from __future__ import annotations

import logging
import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

if os.environ.get("KOE_HEADLESS", "0") == "1":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_OPENGL", "software")

QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

from db.connection import bootstrap_if_needed
from ui.dup_tab import DupTab
from ui.settings_tab import SettingsTab
from ui.tags_tab import TagsTab
from utils.paths import ensure_dirs, get_db_path, migrate_data_dir_if_needed

logger = logging.getLogger(__name__)


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
        self._tabs.addTab(TagsTab(self), "Tags")
        self._tabs.addTab(DupTab(self), "Duplicates")
        self._tabs.addTab(SettingsTab(self), "Settings")
        self.setCentralWidget(self._tabs)


def run() -> None:
    """Launch the kobato-eyes GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
