"""UI for tag-based search in kobato-eyes."""

from __future__ import annotations

from typing import Iterable

from PyQt6.QtWidgets import QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from core.query import translate_query


class TagsTab(QWidget):
    """Provide a search bar and placeholder results for tag queries."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._query_input = QLineEdit(self)
        self._query_input.setPlaceholderText("Search tags…")
        self._search_button = QPushButton("Search", self)
        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)

        self._results = QTableWidget(0, 2, self)
        self._results.setHorizontalHeaderLabels(["SQL WHERE", "Parameters"])
        self._results.horizontalHeader().setStretchLastSection(True)
        self._results.verticalHeader().setVisible(False)
        self._results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        layout = QVBoxLayout(self)
        layout.addWidget(self._query_input)
        layout.addWidget(self._search_button)
        layout.addWidget(self._status_label)
        layout.addWidget(self._results)

        self._search_button.clicked.connect(self._execute_search)
        self._query_input.returnPressed.connect(self._execute_search)

    def _execute_search(self) -> None:
        query = self._query_input.text().strip()
        if not query:
            self._status_label.setText("Enter a query to search tags.")
            self._results.setRowCount(0)
            return

        try:
            fragment = translate_query(query)
        except ValueError as exc:
            self._status_label.setText(str(exc))
            self._results.setRowCount(0)
            return

        self._status_label.clear()
        self._results.setRowCount(1)
        self._results.setItem(0, 0, QTableWidgetItem(fragment.where))
        self._results.setItem(0, 1, QTableWidgetItem(str(fragment.params)))

    def display_results(self, rows: Iterable[tuple[str, list[object]]]) -> None:
        """Populate the result grid with preformatted rows."""
        self._results.setRowCount(0)
        for row_index, (where_stmt, params) in enumerate(rows):
            self._results.insertRow(row_index)
            self._results.setItem(row_index, 0, QTableWidgetItem(where_stmt))
            self._results.setItem(row_index, 1, QTableWidgetItem(str(params)))


__all__ = ["TagsTab"]
