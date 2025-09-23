"""Minimal PyQt6 application entry point for kobato-eyes."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """Main window presenting basic navigation tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("kobato-eyes")
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_placeholder_tab("Tags"), "Tags")
        self._tabs.addTab(self._build_placeholder_tab("Duplicates"), "Duplicates")
        self.setCentralWidget(self._tabs)

    @staticmethod
    def _build_placeholder_tab(name: str) -> QWidget:
        """Create a simple placeholder tab until real content is wired."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(QLabel(f"{name} view coming soon"))
        return container


def run() -> None:
    """Launch the kobato-eyes GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
