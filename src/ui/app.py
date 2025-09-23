"""Minimal PyQt6 application entry point for kobato-eyes."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

from ui.dup_tab import DupTab
from ui.settings_tab import SettingsTab
from ui.tags_tab import TagsTab


class MainWindow(QMainWindow):
    """Main window presenting basic navigation tabs."""

    def __init__(self) -> None:
        super().__init__()
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
