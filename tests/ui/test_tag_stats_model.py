"""Tests for tag statistics table model helpers."""

from __future__ import annotations

import csv
import sqlite3
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest
from PyQt6.QtCore import QEvent, QModelIndex, Qt, QThread
from PyQt6.QtGui import QCloseEvent, QKeyEvent

from ui.tag_stats import (
    TagStatsDialog,
    _load_thresholds,
    _TagStatsModel,
    category_name,
    format_score,
    load_tag_stats_rows,
    merge_thresholds,
    tag_stats_csv_rows,
    write_tag_stats_csv,
)


def _make_stats_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category INTEGER NOT NULL);
        CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT NOT NULL, is_present INTEGER NOT NULL);
        CREATE TABLE file_tags (file_id INTEGER NOT NULL, tag_id INTEGER NOT NULL, score REAL NOT NULL);
        CREATE TABLE tagger_thresholds (category INTEGER PRIMARY KEY, threshold REAL NOT NULL);
        INSERT INTO tags VALUES (1, '1girl', 0), (2, 'kobato', 4), (3, 'low_score', 0);
        INSERT INTO files VALUES (10, 'a.png', 1), (11, 'b.png', 1), (12, 'gone.png', 0);
        INSERT INTO file_tags VALUES
            (10, 1, 0.90),
            (11, 1, 0.50),
            (10, 2, 0.30),
            (11, 3, 0.10),
            (12, 1, 1.00);
        INSERT INTO tagger_thresholds VALUES (0, 0.40);
        """
    )
    return conn


def _make_stats_db(path: Path) -> None:
    """Create a file-backed stats database for worker-thread tests."""

    source = _make_stats_conn()
    try:
        target = sqlite3.connect(path)
        try:
            source.backup(target)
            target.commit()
        finally:
            target.close()
    finally:
        source.close()


def test_load_thresholds_merges_fallbacks_and_database_values() -> None:
    conn = _make_stats_conn()

    thresholds = _load_thresholds(conn)

    assert thresholds[0] == 0.40
    assert thresholds[4] == 0.25
    assert thresholds[1] == 0.0


def test_tag_stats_pure_helpers_format_and_skip_bad_thresholds() -> None:
    thresholds = merge_thresholds([(0, "0.8"), ("bad", "0.5"), (5, object())])

    assert thresholds[0] == 0.8
    assert thresholds[4] == 0.25
    assert thresholds[5] == 0.0
    assert category_name(4) == "character"
    assert category_name(99) == "99"
    assert format_score(0.12345) == "0.123"


def test_tag_stats_model_loads_and_formats_rows() -> None:
    conn = _make_stats_conn()
    model = _TagStatsModel()

    model.load(conn, category=None, respect_thresholds=True)

    assert model.rowCount() == 2
    assert model.columnCount() == 5
    assert model.headerData(0, Qt.Orientation.Horizontal) == "Category"
    assert model.headerData(0, Qt.Orientation.Vertical) == 1
    assert model.rowCount(QModelIndex()) == 2
    assert model.tag_at(0) == "1girl"
    assert model.tag_at(99) == ""

    first = model.index(0, 0)
    assert model.data(first) == "general"
    assert model.data(first, Qt.ItemDataRole.UserRole) == 0
    assert model.data(model.index(0, 2)) == 2
    assert model.data(model.index(0, 3)) == "0.700"
    assert model.data(model.index(0, 4), Qt.ItemDataRole.UserRole) == 0.9
    assert model.data(QModelIndex()) is None


def test_tag_stats_model_filters_category_without_thresholds() -> None:
    conn = _make_stats_conn()
    model = _TagStatsModel()

    model.load(conn, category=4, respect_thresholds=False)

    assert model.rowCount() == 1
    assert model.data(model.index(0, 0)) == "character"
    assert model.data(model.index(0, 1)) == "kobato"


def test_load_tag_stats_rows_supports_unlimited_filter_and_sort() -> None:
    conn = _make_stats_conn()

    rows = load_tag_stats_rows(
        conn,
        category=None,
        respect_thresholds=False,
        filter_text="o",
        limit=None,
        sort_column=1,
        sort_order=Qt.SortOrder.DescendingOrder,
    )

    assert [row[1] for row in rows] == ["low_score", "kobato"]
    assert tag_stats_csv_rows(rows) == [
        ["general", "low_score", 1, "0.100", "0.100"],
        ["character", "kobato", 1, "0.300", "0.300"],
    ]


def test_load_tag_stats_rows_escapes_literal_like_wildcards() -> None:
    conn = _make_stats_conn()
    conn.execute("INSERT INTO tags VALUES (4, 'under_score', 0)")
    conn.execute("INSERT INTO file_tags VALUES (10, 4, 0.80)")

    rows = load_tag_stats_rows(
        conn,
        category=None,
        respect_thresholds=False,
        filter_text="_",
        limit=None,
    )

    assert [row[1] for row in rows] == ["low_score", "under_score"]


def test_load_tag_stats_rows_can_export_beyond_display_limit() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category INTEGER NOT NULL);
        CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT NOT NULL, is_present INTEGER NOT NULL);
        CREATE TABLE file_tags (file_id INTEGER NOT NULL, tag_id INTEGER NOT NULL, score REAL NOT NULL);
        INSERT INTO files VALUES (1, 'a.png', 1);
        """
    )
    conn.executemany("INSERT INTO tags VALUES (?, ?, 0)", ((idx, f"tag_{idx:04d}") for idx in range(1, 1002)))
    conn.executemany("INSERT INTO file_tags VALUES (1, ?, 0.80)", ((idx,) for idx in range(1, 1002)))

    visible_rows = load_tag_stats_rows(conn, category=None, respect_thresholds=False)
    export_rows = load_tag_stats_rows(conn, category=None, respect_thresholds=False, limit=None)

    assert len(visible_rows) == 1000
    assert len(export_rows) == 1001


def test_write_tag_stats_csv_adds_suffix_and_writes_utf8_sig(tmp_path: Path) -> None:
    output = write_tag_stats_csv(
        ["Category", "Tag"],
        [["general", "1girl"], ["character", "kobato"]],
        tmp_path / "stats",
    )

    assert output == tmp_path / "stats.csv"
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["Category", "Tag"],
            ["general", "1girl"],
            ["character", "kobato"],
        ]


@pytest.mark.gui
def test_tag_stats_dialog_filters_and_ignores_selection_without_tags_parent(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    conn = _make_stats_conn()
    dialog = TagStatsDialog(lambda: conn)
    qtbot.addWidget(dialog)
    fake_tags_tab = ModuleType("ui.tags_tab")
    fake_tags_tab.TagsTab = type("TagsTab", (), {})
    monkeypatch.setitem(sys.modules, "ui.tags_tab", fake_tags_tab)

    assert dialog._model.rowCount() == 2
    assert dialog._filter_edit.placeholderText() == "type to filter tags..."
    assert "1000-row display limit" in dialog._export_button.toolTip()
    dialog._filter_edit.setText("kobato")
    assert dialog._proxy.rowCount() == 1

    dialog._table.selectRow(0)
    dialog._apply_selected_tag()


@pytest.mark.gui
def test_tag_stats_dialog_enter_with_no_selection_is_noop(qtbot) -> None:  # type: ignore[no-untyped-def]
    conn = _make_stats_conn()
    dialog = TagStatsDialog(lambda: conn)
    qtbot.addWidget(dialog)
    dialog._table.clearSelection()
    dialog._table.setCurrentIndex(QModelIndex())
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)

    dialog.keyPressEvent(event)

    assert event.isAccepted()


@pytest.mark.gui
def test_tag_stats_dialog_waits_for_running_worker_thread(qtbot) -> None:  # type: ignore[no-untyped-def]
    thread = QThread()
    thread.start()
    qtbot.waitUntil(thread.isRunning, timeout=1000)

    assert TagStatsDialog._wait_for_thread(thread)

    assert not thread.isRunning()


@pytest.mark.gui
def test_tag_stats_dialog_waits_for_all_tracked_worker_threads(qtbot) -> None:  # type: ignore[no-untyped-def]
    conn = _make_stats_conn()
    dialog = TagStatsDialog(lambda: conn)
    qtbot.addWidget(dialog)
    threads = [QThread(), QThread()]
    for thread in threads:
        thread.start()
    qtbot.waitUntil(lambda: all(thread.isRunning() for thread in threads), timeout=1000)
    dialog._load_threads = list(threads)

    assert dialog._wait_for_worker_threads()

    assert all(not thread.isRunning() for thread in threads)


@pytest.mark.gui
def test_tag_stats_dialog_close_ignores_when_worker_does_not_stop(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    conn = _make_stats_conn()
    dialog = TagStatsDialog(lambda: conn)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "_wait_for_worker_threads", lambda: False)
    event = QCloseEvent()

    dialog.closeEvent(event)

    assert not event.isAccepted()
    assert not dialog._loading_widget.isHidden()
    assert dialog._close_pending
    assert dialog._loading_label.text() == "Finishing tag statistics task. This window will close when it completes."
    assert not dialog._loading_bar.isVisible()


@pytest.mark.gui
def test_tag_stats_dialog_async_loads_rows_after_show(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    db_path = tmp_path / "stats.db"
    _make_stats_db(db_path)

    def conn_factory() -> sqlite3.Connection:
        time.sleep(0.05)
        return sqlite3.connect(db_path)

    dialog = TagStatsDialog(conn_factory, async_load=True)
    qtbot.addWidget(dialog)
    assert dialog._load_generation == 0
    dialog.show()

    qtbot.waitUntil(lambda: dialog._loading_widget.isVisible(), timeout=1000)
    qtbot.waitUntil(lambda: dialog._model.rowCount() == 2, timeout=3000)

    assert not dialog._loading_widget.isVisible()
