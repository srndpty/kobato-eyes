"""Headless tests for asynchronous indexing behaviour in TagsTab."""

from __future__ import annotations

import sqlite3
import time
from typing import Iterable
from unittest import mock
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication

from db.schema import apply_schema
from tagger.wd14_onnx import ONNXRUNTIME_MISSING_MESSAGE
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


def test_index_now_async_flow(tags_tab: TagsTab, qapp: QApplication) -> None:
    started = mock.Event()
    allow_finish = mock.Event()

    def _fake_run_index_once(*args, **kwargs):
        started.set()
        allow_finish.wait(2)
        return {
            "scanned": 3,
            "tagged": 2,
            "embedded": 1,
            "elapsed_sec": 0.5,
            "new_or_changed": 2,
            "signatures": 2,
            "hnsw_added": 1,
            "tagger_name": "dummy",
        }

    search_spy = mock.MagicMock()

    with patch("ui.tags_tab.run_index_once", side_effect=_fake_run_index_once):
        with patch.object(tags_tab, "_on_search_clicked", search_spy):
            tags_tab._placeholder_button.click()  # type: ignore[attr-defined]
            assert started.wait(2)
            qapp.processEvents()

            assert not tags_tab._placeholder_button.isEnabled()  # type: ignore[attr-defined]
            assert not tags_tab._search_button.isEnabled()  # type: ignore[attr-defined]
            assert not tags_tab._query_edit.isEnabled()  # type: ignore[attr-defined]

            allow_finish.set()

            for _ in range(100):
                qapp.processEvents()
                if not getattr(tags_tab, "_indexing_active", True):
                    break
                time.sleep(0.01)

    assert search_spy.called
    assert getattr(tags_tab, "_indexing_active", False) is False
    assert tags_tab._status_label.text().startswith("Indexing complete")  # type: ignore[attr-defined]
    assert tags_tab._toast_label.isVisible()  # type: ignore[attr-defined]
    assert (
        tags_tab._toast_label.text()  # type: ignore[attr-defined]
        == "Indexed: 3 files / Tagged: 2 / Embedded: 1 (tagger: dummy)"
    )
    assert tags_tab._placeholder_button.isEnabled()  # type: ignore[attr-defined]


def test_index_now_reports_missing_onnxruntime(tags_tab: TagsTab, qapp: QApplication) -> None:
    with patch("ui.tags_tab.run_index_once", side_effect=RuntimeError(ONNXRUNTIME_MISSING_MESSAGE)):
        tags_tab._placeholder_button.click()  # type: ignore[attr-defined]
        for _ in range(100):
            qapp.processEvents()
            if not getattr(tags_tab, "_indexing_active", True):
                break
            time.sleep(0.01)

    assert tags_tab._toast_label.isVisible()  # type: ignore[attr-defined]
    assert tags_tab._toast_label.text() == ONNXRUNTIME_MISSING_MESSAGE  # type: ignore[attr-defined]
    assert tags_tab._status_label.text() == ONNXRUNTIME_MISSING_MESSAGE  # type: ignore[attr-defined]
    assert tags_tab._placeholder_button.isEnabled()  # type: ignore[attr-defined]
