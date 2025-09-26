"""Dialog for displaying aggregated tag statistics."""

from __future__ import annotations

import sqlite3
from typing import Callable

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    QSortFilterProxyModel,
)
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

_CATEGORY_NAMES: dict[int, str] = {
    0: "general",
    1: "character",
    2: "rating",
    3: "copyright",
    4: "artist",
    5: "meta",
}

_FALLBACK_THRESHOLDS: dict[int, float] = {0: 0.35, 1: 0.25, 3: 0.25}


def _load_thresholds(conn: sqlite3.Connection) -> dict[int, float]:
    """Return tag thresholds using database overrides when available."""

    thresholds: dict[int, float] = {}
    try:
        cursor = conn.execute("SELECT category, threshold FROM tagger_thresholds")
        for category, value in cursor.fetchall():
            try:
                thresholds[int(category)] = float(value)
            except Exception:
                continue
    except sqlite3.Error:
        pass

    merged: dict[int, float] = {key: 0.0 for key in range(6)}
    merged.update(_FALLBACK_THRESHOLDS)
    merged.update(thresholds)
    return merged


class _TagStatsModel(QAbstractTableModel):
    """Table model containing tag statistics."""

    COLUMNS = ("Category", "Tag", "Files", "AvgScore", "MaxScore")

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[tuple[int, str, int, float, float]] = []

    def load(
        self,
        conn: sqlite3.Connection,
        category: int | None,
        respect_thresholds: bool,
    ) -> None:
        """Populate the model using the provided filters."""

        self.beginResetModel()
        self._rows.clear()

        params: list[object] = []
        where_clauses: list[str] = []
        thresholds: dict[int, float] | None = None
        if category is not None:
            where_clauses.append("t.category = ?")
            params.append(int(category))
        if respect_thresholds:
            thresholds = _load_thresholds(conn)

        score_guard = ""
        if thresholds is not None:
            cases = []
            case_params: list[object] = []
            for cat_id in sorted(thresholds):
                cases.append("WHEN ? THEN ?")
                case_params.extend([cat_id, thresholds[cat_id]])
            score_guard = (
                "AND ft.score >= CASE t.category "
                + " ".join(cases)
                + " ELSE 0.0 END "
            )
            params.extend(case_params)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        sql = (
            "SELECT t.category, t.name, COUNT(DISTINCT ft.file_id) AS files, "
            "AVG(ft.score) AS avg_score, MAX(ft.score) AS max_score "
            "FROM tags t "
            "JOIN file_tags ft ON ft.tag_id = t.id "
            f"{where_sql} "
            f"{'AND ' if where_sql else 'WHERE '}1 = 1 {score_guard}"
            "GROUP BY t.id "
            "ORDER BY files DESC, t.name ASC"
        )

        for row in conn.execute(sql, params):
            category_id = int(row[0])
            tag_name = str(row[1])
            file_count = int(row[2])
            avg_score = float(row[3]) if row[3] is not None else 0.0
            max_score = float(row[4]) if row[4] is not None else 0.0
            self._rows.append((category_id, tag_name, file_count, avg_score, max_score))

        self.endResetModel()

    # Qt model API ---------------------------------------------------------
    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        return len(self.COLUMNS)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return section + 1

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        row_index = index.row()
        column_index = index.column()
        category_id, tag_name, file_count, avg_score, max_score = self._rows[row_index]

        if role == Qt.ItemDataRole.DisplayRole:
            if column_index == 0:
                return _CATEGORY_NAMES.get(category_id, str(category_id))
            if column_index == 1:
                return tag_name
            if column_index == 2:
                return file_count
            if column_index == 3:
                return f"{avg_score:.3f}"
            if column_index == 4:
                return f"{max_score:.3f}"
        if role == Qt.ItemDataRole.UserRole:
            if column_index == 0:
                return category_id
            if column_index == 1:
                return tag_name
            if column_index == 2:
                return file_count
            if column_index == 3:
                return avg_score
            if column_index == 4:
                return max_score
        return None

    def tag_at(self, row: int) -> str:
        """Return the tag name for the requested row."""

        if 0 <= row < len(self._rows):
            return self._rows[row][1]
        return ""


class TagStatsDialog(QDialog):
    """Modal dialog showing aggregate statistics for tags."""

    def __init__(
        self,
        conn_factory: Callable[[], sqlite3.Connection],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tag Stats")
        self.resize(800, 600)

        self._conn_factory = conn_factory

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox(self)
        self._category_combo.addItem("All", None)
        for category_id in range(6):
            name = _CATEGORY_NAMES.get(category_id, str(category_id))
            self._category_combo.addItem(name.title(), category_id)
        top_bar.addWidget(self._category_combo)

        self._threshold_check = QCheckBox("Respect thresholds", self)
        self._threshold_check.setChecked(True)
        top_bar.addWidget(self._threshold_check)

        top_bar.addStretch(1)
        top_bar.addWidget(QLabel("Filter:"))
        self._filter_edit = QLineEdit(self)
        self._filter_edit.setPlaceholderText("type to filter tags…")
        top_bar.addWidget(self._filter_edit)

        self._model = _TagStatsModel()
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._proxy.setFilterKeyColumn(1)
        self._proxy.setSortRole(Qt.ItemDataRole.UserRole)

        self._table = QTableView(self)
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.doubleClicked.connect(self._apply_selected_tag)
        self._table.activated.connect(self._apply_selected_tag)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self._table)

        self._category_combo.currentIndexChanged.connect(self._reload)
        self._threshold_check.toggled.connect(self._reload)
        self._filter_edit.textChanged.connect(self._proxy.setFilterFixedString)

        self._reload()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: D401 - Qt signature
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply_selected_tag()
            event.accept()
            return
        super().keyPressEvent(event)

    def _reload(self) -> None:
        with self._conn_factory() as conn:
            category = self._category_combo.currentData()
            respect = self._threshold_check.isChecked()
            self._model.load(conn, category, respect)
        self._table.resizeColumnsToContents()

    def _apply_selected_tag(self) -> None:
        index = self._table.currentIndex()
        if not index.isValid():
            return
        source_row = self._proxy.mapToSource(index).row()
        tag = self._model.tag_at(source_row)

        from ui.tags_tab import TagsTab

        parent = self.parent()
        if isinstance(parent, TagsTab):
            edit = parent._query_edit
            current_text = edit.text().strip()
            if current_text:
                new_text = f"{current_text} {tag} "
            else:
                new_text = f"{tag} "
            blocked = edit.blockSignals(True)
            try:
                edit.setText(new_text)
            finally:
                edit.blockSignals(blocked)
            edit.setCursorPosition(len(new_text))


__all__ = ["TagStatsDialog"]

