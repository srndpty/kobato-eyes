"""Smoke tests for TagsTab UI behaviour."""

from __future__ import annotations

import sqlite3
from typing import Iterable
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication

from db.schema import apply_schema
from ui.tags_tab import TagsTab

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Iterable[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def tags_tab(qapp: QApplication):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    with patch("ui.tags_tab.get_conn", return_value=conn):
        widget = TagsTab()
        try:
            yield widget
        finally:
            widget.deleteLater()
            conn.close()


def test_toggle_views(tags_tab: TagsTab) -> None:
    tags_tab._stack.setCurrentWidget(tags_tab._table_view)  # type: ignore[attr-defined]
    tags_tab._table_button.setChecked(True)  # type: ignore[attr-defined]
    assert tags_tab._stack.currentWidget() is tags_tab._table_view  # type: ignore[attr-defined]
    tags_tab._grid_button.setChecked(True)  # type: ignore[attr-defined]
    assert tags_tab._stack.currentWidget() is tags_tab._grid_view  # type: ignore[attr-defined]
    tags_tab._table_button.setChecked(True)  # type: ignore[attr-defined]
    assert tags_tab._stack.currentWidget() is tags_tab._table_view  # type: ignore[attr-defined]


def test_index_now_triggers_pipeline(tags_tab: TagsTab) -> None:
    with patch("ui.tags_tab.run_index_once") as mocked:
        tags_tab._placeholder_button.click()  # type: ignore[attr-defined]
        mocked.assert_called_once()
