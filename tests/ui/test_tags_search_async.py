from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
from unittest import mock

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication

from core.config import PipelineSettings
from db.schema import apply_schema
from ui.tags_tab import TagsTab
from ui.viewmodels import TagsViewModel

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Iterable[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def populated_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    tag_ids: list[int] = []
    for idx in range(3):
        cur = conn.execute("INSERT INTO tags (name, category) VALUES (?, ?)", (f"tag{idx}", 0))
        tag_ids.append(int(cur.lastrowid))
    for file_idx in range(6):
        cur = conn.execute(
            "INSERT INTO files (path, size, mtime, is_present, width, height) VALUES (?, ?, ?, 1, ?, ?)",
            (f"/tmp/file_{file_idx}.png", 1024 + file_idx, float(file_idx + 0.1), 64, 64),
        )
        file_id = int(cur.lastrowid)
        for tag_id in tag_ids:
            conn.execute(
                "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                (file_id, tag_id, 0.9),
            )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def tags_tab(qapp: QApplication, populated_db: Path) -> Iterable[TagsTab]:
    def connection_factory(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    vm = TagsViewModel(db_path=populated_db, connection_factory=connection_factory)
    vm.ensure_directories = lambda: None  # type: ignore[assignment]
    vm.load_settings = mock.MagicMock(return_value=PipelineSettings(roots=[str(populated_db.parent)]))  # type: ignore[assignment]
    vm.translate_query = mock.MagicMock(return_value=SimpleNamespace(where="1=1", params=[]))  # type: ignore[assignment]
    vm.extract_positive_terms = mock.MagicMock(return_value=set())  # type: ignore[assignment]
    vm.list_tag_names = mock.MagicMock(return_value=[])  # type: ignore[assignment]
    vm.iter_paths_for_search = mock.MagicMock(return_value=[])  # type: ignore[assignment]

    widget = TagsTab(
        view_model=vm,
        search_chunk_size=1,
        search_chunk_delay=0.01,
    )
    widget._bootstrap_results_if_any = lambda: None  # type: ignore[attr-defined]
    try:
        yield widget
    finally:
        widget.deleteLater()


def _wait_for(condition, qapp: QApplication, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def _await_idle(tab: TagsTab, qapp: QApplication, timeout: float = 3.0) -> None:
    assert _wait_for(lambda: not tab._search_busy and tab._search_worker is None, qapp, timeout)


def test_async_search_streams_results(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._search_timer.stop()
    tags_tab._cancel_active_search()
    _await_idle(tags_tab, qapp)

    tags_tab._query_edit.setText("test")  # type: ignore[attr-defined]
    tags_tab._on_search_clicked()

    row_counts: list[int] = []

    def record_count() -> None:
        count = tags_tab._table_model.rowCount()  # type: ignore[attr-defined]
        if not row_counts or row_counts[-1] != count:
            row_counts.append(count)

    assert _wait_for(
        lambda: record_count() or (not tags_tab._search_busy and tags_tab._search_worker is None),
        qapp,
        timeout=4.0,
    )
    _await_idle(tags_tab, qapp)
    record_count()

    assert row_counts[-1] == 6
    assert len([value for value in row_counts if value]) >= 2


def test_async_search_cancel(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._query_edit.setText("again")  # type: ignore[attr-defined]
    tags_tab._on_search_clicked()
    assert _wait_for(lambda: tags_tab._table_model.rowCount() > 0, qapp, timeout=2.0)  # type: ignore[attr-defined]
    tags_tab._cancel_active_search()
    _await_idle(tags_tab, qapp)
    assert tags_tab._last_search_cancelled


def test_async_search_indeterminate_when_count_disabled(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._search_count_timeout = 0.0
    tags_tab._query_edit.setText("no-count")  # type: ignore[attr-defined]
    tags_tab._on_search_clicked()
    assert _wait_for(lambda: tags_tab._table_model.rowCount() > 0, qapp, timeout=2.0)  # type: ignore[attr-defined]
    assert not tags_tab._search_overlay.is_determinate()
    _await_idle(tags_tab, qapp)
    assert not tags_tab._search_overlay.isVisible()
