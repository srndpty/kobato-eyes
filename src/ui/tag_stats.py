"""Dialog for displaying aggregated tag statistics."""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from typing import Any, Callable, cast

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
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
    QProgressBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

_CATEGORY_NAMES: dict[int, str] = {
    0: "general",
    1: "artist",
    2: "rating",
    3: "copyright",
    4: "character",
    5: "meta",
}

_FALLBACK_THRESHOLDS: dict[int, float] = {0: 0.35, 4: 0.25, 3: 0.25}
_TagStatsRow = tuple[int, str, int, float, float]


def category_name(category_id: int) -> str:
    """Return the display name for a Danbooru category id."""

    return _CATEGORY_NAMES.get(category_id, str(category_id))


def format_score(value: float) -> str:
    """Return a tag score formatted for the stats table."""

    return f"{float(value):.3f}"


def merge_thresholds(rows: list[tuple[object, object]]) -> dict[int, float]:
    """Merge threshold rows with built-in fallback thresholds."""

    thresholds: dict[int, float] = {}
    for category, value in rows:
        try:
            thresholds[int(cast(Any, category))] = float(cast(Any, value))
        except (TypeError, ValueError):
            continue
    merged: dict[int, float] = {key: 0.0 for key in range(6)}
    merged.update(_FALLBACK_THRESHOLDS)
    merged.update(thresholds)
    return merged


def _load_thresholds(conn: sqlite3.Connection) -> dict[int, float]:
    """Return tag thresholds using database overrides when available."""

    try:
        cursor = conn.execute("SELECT category, threshold FROM tagger_thresholds")
        rows = list(cursor.fetchall())
    except sqlite3.Error:
        rows = []
    return merge_thresholds(rows)


def load_tag_stats_rows(
    conn: sqlite3.Connection,
    category: int | None,
    respect_thresholds: bool,
) -> list[_TagStatsRow]:
    """Return aggregated tag statistics rows for the requested filters."""

    params: list[object] = []
    where_conditions: list[str] = []
    thresholds: dict[int, float] | None = None
    if category is not None:
        where_conditions.append("t.category = ?")
        params.append(int(category))
    if respect_thresholds:
        thresholds = _load_thresholds(conn)

    score_condition = ""
    if thresholds is not None:
        cases = []
        case_params: list[object] = []
        for cat_id in sorted(thresholds):
            cases.append("WHEN ? THEN ?")
            case_params.extend([cat_id, thresholds[cat_id]])
        score_condition = "ft.score >= CASE t.category " + " ".join(cases) + " ELSE 0.0 END"
        params.extend(case_params)

    where_conditions.append("f.is_present = 1")
    if score_condition:
        where_conditions.append(score_condition)

    where_sql = "WHERE " + " AND ".join(where_conditions)
    sql = (
        "SELECT t.category, t.name, COUNT(DISTINCT ft.file_id) AS files, "
        "AVG(ft.score) AS avg_score, MAX(ft.score) AS max_score "
        "FROM tags t "
        "JOIN file_tags ft ON ft.tag_id = t.id "
        "JOIN files f ON f.id = ft.file_id "
        f"{where_sql} "
        "GROUP BY t.id "
        "ORDER BY files DESC, t.name ASC "
        "LIMIT 1000"
    )

    rows: list[_TagStatsRow] = []
    for row in conn.execute(sql, params):
        category_id = int(row[0])
        tag_name = str(row[1])
        file_count = int(row[2])
        avg_score = float(row[3]) if row[3] is not None else 0.0
        max_score = float(row[4]) if row[4] is not None else 0.0
        rows.append((category_id, tag_name, file_count, avg_score, max_score))
    return rows


class _StatsLoadWorker(QObject):
    """Load tag statistics on a worker thread."""

    loaded = pyqtSignal(int, list)
    error = pyqtSignal(int, str)

    def __init__(
        self,
        generation: int,
        conn_factory: Callable[[], sqlite3.Connection],
        category: int | None,
        respect_thresholds: bool,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._conn_factory = conn_factory
        self._category = category
        self._respect_thresholds = respect_thresholds

    @pyqtSlot()
    def run(self) -> None:
        """Execute the statistics query and emit the loaded rows."""

        conn: sqlite3.Connection | None = None
        try:
            conn = self._conn_factory()
            rows = load_tag_stats_rows(conn, self._category, self._respect_thresholds)
        except Exception as exc:  # pragma: no cover - UI surfaces the message
            self.error.emit(self._generation, str(exc))
        else:
            self.loaded.emit(self._generation, rows)
        finally:
            if conn is not None:
                with suppress(Exception):
                    conn.close()
            thread = QThread.currentThread()
            if thread is not None:
                thread.quit()


class _TagStatsModel(QAbstractTableModel):
    """Table model containing tag statistics."""

    COLUMNS = ("Category", "Tag", "Files", "AvgScore", "MaxScore")

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[_TagStatsRow] = []

    def load(
        self,
        conn: sqlite3.Connection,
        category: int | None,
        respect_thresholds: bool,
        # limit: int,
    ) -> None:
        """Populate the model using the provided filters."""

        self.set_rows(load_tag_stats_rows(conn, category, respect_thresholds))

    def set_rows(self, rows: list[_TagStatsRow]) -> None:
        """Replace the model rows on the GUI thread."""

        self.beginResetModel()
        self._rows = list(rows)
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
                return category_name(category_id)
            if column_index == 1:
                return tag_name
            if column_index == 2:
                return file_count
            if column_index == 3:
                return format_score(avg_score)
            if column_index == 4:
                return format_score(max_score)
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
        *,
        async_load: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tag Stats")
        self.resize(800, 600)

        self._conn_factory = conn_factory
        self._async_load = async_load
        self._load_generation = 0
        self._load_thread: QThread | None = None
        self._load_worker: _StatsLoadWorker | None = None

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
        header = self._table.verticalHeader()
        if header is not None:
            header.setVisible(False)
        self._table.doubleClicked.connect(self._apply_selected_tag)
        self._table.activated.connect(self._apply_selected_tag)

        self._loading_bar = QProgressBar(self)
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_label = QLabel("Loading tag statistics...", self)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_widget = QWidget(self)
        loading_layout = QHBoxLayout(self._loading_widget)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        loading_layout.addWidget(self._loading_label)
        loading_layout.addWidget(self._loading_bar)
        self._loading_widget.hide()

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self._loading_widget)
        layout.addWidget(self._table)

        self._category_combo.currentIndexChanged.connect(self._reload)
        self._threshold_check.toggled.connect(self._reload)
        self._filter_edit.textChanged.connect(self._proxy.setFilterFixedString)

        self._reload()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: D401 - Qt signature
        if event is None:
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply_selected_tag()
            event.accept()
            return
        super().keyPressEvent(event)

    def _reload(self) -> None:
        if self._async_load:
            self._reload_async()
            return
        with self._conn_factory() as conn:
            category = self._category_combo.currentData()
            respect = self._threshold_check.isChecked()
            self._model.load(conn, category, respect)
        self._table.resizeColumnsToContents()

    def _reload_async(self) -> None:
        """Start a background statistics load for the current filters."""

        self._load_generation += 1
        generation = self._load_generation
        category = self._category_combo.currentData()
        respect = self._threshold_check.isChecked()
        self._set_loading(True, "Loading tag statistics...")

        thread = QThread(self)
        worker = _StatsLoadWorker(generation, self._conn_factory, category, respect)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._handle_async_loaded)
        worker.error.connect(self._handle_async_error)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_async_refs(thread))
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    def _handle_async_loaded(self, generation: int, rows: list[_TagStatsRow]) -> None:
        """Apply asynchronously loaded rows if they match the latest request."""

        if generation != self._load_generation:
            return
        self._model.set_rows(rows)
        self._table.resizeColumnsToContents()
        self._set_loading(False)

    def _handle_async_error(self, generation: int, message: str) -> None:
        """Show asynchronous load errors without closing the dialog."""

        if generation != self._load_generation:
            return
        self._model.set_rows([])
        self._loading_label.setText(f"Failed to load tag statistics: {message}")
        self._loading_bar.hide()
        self._loading_widget.show()
        self._category_combo.setEnabled(True)
        self._threshold_check.setEnabled(True)

    def _clear_async_refs(self, thread: QThread) -> None:
        """Drop completed worker references."""

        if self._load_thread is thread:
            self._load_thread = None
            self._load_worker = None

    def _set_loading(self, loading: bool, message: str = "") -> None:
        """Update loading controls for asynchronous statistics reads."""

        self._loading_label.setText(message)
        self._loading_bar.setVisible(loading and message.startswith("Loading"))
        self._loading_widget.setVisible(loading)
        self._category_combo.setEnabled(not loading)
        self._threshold_check.setEnabled(not loading)

    def showEvent(self, event) -> None:  # type: ignore[override]
        """Kick asynchronous loading after the dialog has been painted."""

        super().showEvent(event)
        if self._async_load and self._load_generation == 0:
            QTimer.singleShot(0, self._reload)

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


__all__ = ["TagStatsDialog", "category_name", "format_score", "merge_thresholds"]
