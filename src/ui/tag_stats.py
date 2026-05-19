"""Dialog for displaying aggregated tag statistics."""

from __future__ import annotations

import csv
import logging
import sqlite3
from contextlib import suppress
from pathlib import Path
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
from PyQt6.QtGui import QCloseEvent, QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
_CsvRow = list[object]
_DISPLAY_ROW_LIMIT = 1000
_THREAD_WAIT_TIMEOUT_MS = 3000
_SORT_EXPRESSIONS = {
    0: "t.category",
    1: "t.name",
    2: "files",
    3: "avg_score",
    4: "max_score",
}

logger = logging.getLogger(__name__)


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


def _escape_like_pattern(text: str) -> str:
    """Escape a literal value for a SQLite LIKE pattern."""

    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sort_clause(column: int, order: Qt.SortOrder) -> str:
    """Return a safe SQL ORDER BY clause matching a table sort column."""

    expression = _SORT_EXPRESSIONS.get(column, "files")
    direction = "ASC" if order == Qt.SortOrder.AscendingOrder else "DESC"
    tie_breaker = ", t.name ASC" if expression != "t.name" else ""
    return f"ORDER BY {expression} {direction}{tie_breaker}"


def tag_stats_csv_rows(rows: list[_TagStatsRow]) -> list[_CsvRow]:
    """Convert raw statistics rows to the display-like CSV representation."""

    return [
        [category_name(category_id), tag_name, file_count, format_score(avg_score), format_score(max_score)]
        for category_id, tag_name, file_count, avg_score, max_score in rows
    ]


def write_tag_stats_csv(headers: list[str], rows: list[_CsvRow], file_path: str | Path) -> Path:
    """Write tag statistics rows to a UTF-8 CSV file and return its path."""

    path = Path(file_path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def load_tag_stats_rows(
    conn: sqlite3.Connection,
    category: int | None,
    respect_thresholds: bool,
    *,
    filter_text: str = "",
    limit: int | None = _DISPLAY_ROW_LIMIT,
    sort_column: int = 2,
    sort_order: Qt.SortOrder = Qt.SortOrder.DescendingOrder,
) -> list[_TagStatsRow]:
    """Return aggregated tag statistics rows for the requested filters."""

    params: list[object] = []
    where_conditions: list[str] = []
    thresholds: dict[int, float] | None = None
    if category is not None:
        where_conditions.append("t.category = ?")
        params.append(int(category))
    if filter_text:
        where_conditions.append("LOWER(t.name) LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like_pattern(filter_text.lower())}%")
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
    order_sql = _sort_clause(sort_column, sort_order)
    limit_sql = "" if limit is None else "LIMIT ?"
    if limit is not None:
        params.append(max(0, int(limit)))

    sql = (
        "SELECT t.category, t.name, COUNT(DISTINCT ft.file_id) AS files, "
        "AVG(ft.score) AS avg_score, MAX(ft.score) AS max_score "
        "FROM tags t "
        "JOIN file_tags ft ON ft.tag_id = t.id "
        "JOIN files f ON f.id = ft.file_id "
        f"{where_sql} "
        "GROUP BY t.id "
        f"{order_sql} "
        f"{limit_sql}"
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


class _StatsExportWorker(QObject):
    """Export complete tag statistics on a worker thread."""

    exported = pyqtSignal(int, str, int)
    error = pyqtSignal(int, str)

    def __init__(
        self,
        generation: int,
        conn_factory: Callable[[], sqlite3.Connection],
        file_path: str | Path,
        category: int | None,
        respect_thresholds: bool,
        filter_text: str,
        sort_column: int,
        sort_order: Qt.SortOrder,
        headers: list[str],
    ) -> None:
        super().__init__()
        self._generation = generation
        self._conn_factory = conn_factory
        self._file_path = file_path
        self._category = category
        self._respect_thresholds = respect_thresholds
        self._filter_text = filter_text
        self._sort_column = sort_column
        self._sort_order = sort_order
        self._headers = headers

    @pyqtSlot()
    def run(self) -> None:
        """Query all matching rows and write them to CSV."""

        conn: sqlite3.Connection | None = None
        try:
            conn = self._conn_factory()
            rows = load_tag_stats_rows(
                conn,
                self._category,
                self._respect_thresholds,
                filter_text=self._filter_text,
                limit=None,
                sort_column=self._sort_column,
                sort_order=self._sort_order,
            )
            if not rows:
                self.exported.emit(self._generation, "", 0)
                return
            csv_rows = tag_stats_csv_rows(rows)
            path = write_tag_stats_csv(self._headers, csv_rows, self._file_path)
        except Exception as exc:  # pragma: no cover - UI surfaces the message
            self.error.emit(self._generation, str(exc))
        else:
            self.exported.emit(self._generation, str(path), len(rows))
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

        # The factory is called from non-GUI worker threads for async load/export.
        self._conn_factory = conn_factory
        self._async_load = async_load
        self._load_generation = 0
        self._load_thread: QThread | None = None
        self._load_worker: _StatsLoadWorker | None = None
        self._load_threads: list[QThread] = []
        self._export_generation = 0
        self._export_thread: QThread | None = None
        self._export_worker: _StatsExportWorker | None = None
        self._export_threads: list[QThread] = []

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
        self._export_button = QPushButton("Export CSV", self)
        self._export_button.setToolTip("Export all tag statistics matching the current filters")
        top_bar.addWidget(self._export_button)

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
        self._filter_edit.textChanged.connect(lambda: self._update_export_button())
        self._export_button.clicked.connect(self._on_export_csv)

        if not self._async_load:
            self._reload()
        else:
            self._export_button.setEnabled(False)

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
        self._update_export_button()

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
        worker.loaded.connect(self._handle_load_finished)
        worker.error.connect(self._handle_load_error)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_async_refs(thread))
        self._load_thread = thread
        self._load_worker = worker
        self._load_threads.append(thread)
        thread.start()

    def _handle_load_finished(self, generation: int, rows: list[_TagStatsRow]) -> None:
        """Apply asynchronously loaded rows if they match the latest request."""

        if generation != self._load_generation:
            return
        self._model.set_rows(rows)
        self._table.resizeColumnsToContents()
        self._set_loading(False)
        self._update_export_button()

    def _handle_load_error(self, generation: int, message: str) -> None:
        """Show asynchronous load errors without closing the dialog."""

        if generation != self._load_generation:
            return
        self._model.set_rows([])
        self._loading_label.setText(f"Failed to load tag statistics: {message}")
        self._loading_bar.hide()
        self._loading_widget.show()
        self._category_combo.setEnabled(True)
        self._threshold_check.setEnabled(True)
        self._filter_edit.setEnabled(True)
        self._export_button.setEnabled(False)

    def _clear_async_refs(self, thread: QThread) -> None:
        """Drop completed worker references."""

        if self._load_thread is thread:
            self._load_thread = None
            self._load_worker = None
        with suppress(ValueError):
            self._load_threads.remove(thread)

    def _clear_export_refs(self, thread: QThread) -> None:
        """Drop completed export worker references."""

        if self._export_thread is thread:
            self._export_thread = None
            self._export_worker = None
        with suppress(ValueError):
            self._export_threads.remove(thread)

    def _set_loading(self, loading: bool, message: str = "") -> None:
        """Update loading controls for asynchronous statistics reads."""

        self._loading_label.setText(message)
        self._loading_bar.setVisible(loading and (message.startswith("Loading") or message.startswith("Exporting")))
        self._loading_widget.setVisible(loading)
        self._category_combo.setEnabled(not loading)
        self._threshold_check.setEnabled(not loading)
        self._filter_edit.setEnabled(not loading)
        self._export_button.setEnabled(not loading and self._model.rowCount() > 0)

    def _update_export_button(self) -> None:
        """Enable export when the loaded category has data to query."""

        self._export_button.setEnabled(not self._loading_widget.isVisible() and self._model.rowCount() > 0)

    def showEvent(self, event) -> None:  # type: ignore[override]
        """Kick asynchronous loading after the dialog has been painted."""

        super().showEvent(event)
        if self._async_load and self._load_generation == 0:
            QTimer.singleShot(0, self._reload)

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: D401 - Qt signature
        """Wait for running worker threads before the dialog is destroyed."""

        if not self._wait_for_worker_threads():
            if event is not None:
                event.ignore()
            self._set_loading(True, "Finishing tag statistics task...")
            return
        if event is not None:
            super().closeEvent(event)

    def _wait_for_worker_threads(self) -> bool:
        """Return whether active stats worker threads finished before close."""

        finished = True
        finished = self._wait_for_thread(self._load_thread) and finished
        finished = self._wait_for_thread(self._export_thread) and finished
        for thread in list(self._load_threads):
            finished = self._wait_for_thread(thread) and finished
        for thread in list(self._export_threads):
            finished = self._wait_for_thread(thread) and finished
        return finished

    @staticmethod
    def _wait_for_thread(thread: QThread | None) -> bool:
        """Return whether a worker thread is stopped or stopped before timeout."""

        if thread is None or not thread.isRunning():
            return True
        if QThread.currentThread() is thread:
            return False
        thread.quit()
        if not thread.wait(_THREAD_WAIT_TIMEOUT_MS):
            logger.warning("Timed out waiting for tag stats worker thread to finish")
            return False
        return True

    def _on_export_csv(self) -> None:
        """Prompt for a CSV path and export all rows matching current filters."""

        headers = self._csv_headers()
        if self._model.rowCount() <= 0:
            QMessageBox.information(self, "Export tag stats", "No tag statistics to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export tag statistics",
            self._default_export_name(),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        self._start_csv_export(file_path, headers)

    def _start_csv_export(self, file_path: str, headers: list[str]) -> None:
        """Start a background export using the current UI filters."""

        self._export_generation += 1
        generation = self._export_generation
        header = self._table.horizontalHeader()
        sort_column = header.sortIndicatorSection() if header is not None else 2
        sort_order = header.sortIndicatorOrder() if header is not None else Qt.SortOrder.DescendingOrder

        thread = QThread(self)
        worker = _StatsExportWorker(
            generation,
            self._conn_factory,
            file_path,
            self._category_combo.currentData(),
            self._threshold_check.isChecked(),
            self._filter_edit.text(),
            sort_column,
            sort_order,
            headers,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.exported.connect(self._handle_export_finished)
        worker.error.connect(self._handle_export_error)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_export_refs(thread))
        self._export_thread = thread
        self._export_worker = worker
        self._export_threads.append(thread)
        self._set_loading(True, "Exporting tag statistics...")
        thread.start()

    def _handle_export_finished(self, generation: int, file_path: str, row_count: int) -> None:
        """Show the export completion message."""

        if generation != self._export_generation:
            return
        self._set_loading(False)
        if row_count <= 0:
            QMessageBox.information(self, "Export tag stats", "No tag statistics match the current filters.")
            return
        QMessageBox.information(
            self,
            "Export tag stats",
            f"Exported {row_count} matching tag statistics row(s) to {file_path}.",
        )

    def _handle_export_error(self, generation: int, message: str) -> None:
        """Show export errors and restore controls."""

        if generation != self._export_generation:
            return
        self._set_loading(False)
        QMessageBox.warning(self, "Export tag stats", f"Failed to export CSV: {message}")

    def _csv_headers(self) -> list[str]:
        """Return CSV headers matching the stats table."""

        return [
            str(self._model.headerData(column, Qt.Orientation.Horizontal))
            for column in range(self._model.columnCount())
        ]

    def _default_export_name(self) -> str:
        """Return a useful default CSV file name for the selected category."""

        category = self._category_combo.currentData()
        category_part = "all" if category is None else category_name(int(category))
        return f"tag_stats_{category_part}.csv"

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
