"""Smoke tests for TagsTab UI behaviour."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable
from unittest import mock
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication

from core.config import PipelineSettings
from db.schema import apply_schema
from ui.tags_tab import TagsTab

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Iterable[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def tags_tab(qapp: QApplication):
    connections: list[sqlite3.Connection] = []

    def _make_connection(*args, **kwargs) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        connections.append(conn)
        return conn

    with patch("ui.tags_tab.get_conn", side_effect=_make_connection):
        widget = TagsTab()
        widget._view_model.load_settings = mock.MagicMock(  # type: ignore[attr-defined]
            return_value=PipelineSettings(roots=[str(Path.cwd())])
        )
        widget.show()
        qapp.processEvents()
        try:
            yield widget
        finally:
            widget.deleteLater()
            for conn in connections:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass


def test_toggle_views(tags_tab: TagsTab) -> None:
    tags_tab._stack.setCurrentWidget(tags_tab._table_view)  # type: ignore[attr-defined]
    tags_tab._table_button.setChecked(True)  # type: ignore[attr-defined]
    assert tags_tab._stack.currentWidget() is tags_tab._table_view  # type: ignore[attr-defined]
    tags_tab._grid_button.setChecked(True)  # type: ignore[attr-defined]
    assert tags_tab._stack.currentWidget() is tags_tab._grid_view  # type: ignore[attr-defined]
    tags_tab._table_button.setChecked(True)  # type: ignore[attr-defined]
    assert tags_tab._stack.currentWidget() is tags_tab._table_view  # type: ignore[attr-defined]


def test_index_now_triggers_pipeline(tags_tab: TagsTab, qapp: QApplication) -> None:
    done = threading.Event()

    def _fake_run_index_once(*args, **kwargs):
        done.set()
        return {
            "scanned": 1,
            "tagged": 1,
            "elapsed_sec": 0.1,
            "new_or_changed": 1,
            "signatures": 1,
            "tagger_name": "dummy",
        }

    with patch("ui.tags_tab.run_index_once", side_effect=_fake_run_index_once) as mocked:
        tags_tab._placeholder_button.click()  # type: ignore[attr-defined]
        for _ in range(100):
            qapp.processEvents()
            if done.wait(0.01):
                break
            time.sleep(0.01)
        assert done.is_set()
        mocked.assert_called_once()
