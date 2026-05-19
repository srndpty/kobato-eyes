"""Tests for tag statistics table model helpers."""

from __future__ import annotations

import sqlite3
import sys
from types import ModuleType

import pytest
from PyQt6.QtCore import QEvent, QModelIndex, Qt
from PyQt6.QtGui import QKeyEvent

from ui.tag_stats import TagStatsDialog, _load_thresholds, _TagStatsModel, category_name, format_score, merge_thresholds


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
