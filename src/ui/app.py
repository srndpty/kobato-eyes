from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget

from ui.tags_tab import TagsTab

"""Minimal PyQt6 application entry point for kobato-eyes."""


class MainWindow(QMainWindow):
    """Main window presenting basic navigation tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("kobato-eyes")
        self._tabs = QTabWidget()
        self._tabs.addTab(TagsTab(self), "Tags")
        self._tabs.addTab(self._build_placeholder_tab("Duplicates"), "Duplicates")
        self.setCentralWidget(self._tabs)

    @staticmethod
    def _build_placeholder_tab(name: str) -> QWidget:
        """Create a simple placeholder tab until real content is wired."""
        return QWidget()


def run() -> None:
    """Launch the kobato-eyes GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
