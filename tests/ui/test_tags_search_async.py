from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
from unittest import mock

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)
from PyQt6.QtCore import QItemSelectionModel, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.config import PipelineSettings
from db.schema import apply_schema
from tagger.labels_util import TagMeta
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
        image_path = tmp_path / f"file_{file_idx}.png"
        image_path.write_bytes(b"fake image")
        cur = conn.execute(
            "INSERT INTO files (path, size, mtime, is_present, width, height) VALUES (?, ?, ?, 1, ?, ?)",
            (str(image_path), 1024 + file_idx, float(file_idx + 0.1), 64, 64),
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


def test_async_search_returns_first_page_and_loads_more(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._query_edit.setText("test")  # type: ignore[attr-defined]
    tags_tab._on_search_clicked()

    assert _wait_for(lambda: not tags_tab._search_busy and tags_tab._search_worker is None, qapp, timeout=4.0)
    assert tags_tab._table_model.rowCount() == tags_tab._search_chunk_size  # type: ignore[attr-defined]
    assert tags_tab._offset == tags_tab._search_chunk_size  # type: ignore[attr-defined]
    assert tags_tab._can_load_more  # type: ignore[attr-defined]
    assert tags_tab._status_label.text() == "Showing 1 result(s) for 'test'"  # type: ignore[attr-defined]

    tags_tab._on_load_more_clicked()
    assert _wait_for(lambda: not tags_tab._search_busy and tags_tab._search_worker is None, qapp, timeout=4.0)
    assert tags_tab._table_model.rowCount() == 2  # type: ignore[attr-defined]
    assert tags_tab._offset == 2  # type: ignore[attr-defined]
    assert tags_tab._status_label.text() == "Showing 2 result(s) for 'test'"  # type: ignore[attr-defined]
    assert tags_tab._can_load_more  # type: ignore[attr-defined]


def test_new_search_scrolls_results_to_top_but_load_more_keeps_position(
    tags_tab: TagsTab,
    qapp: QApplication,
) -> None:
    _await_idle(tags_tab, qapp)
    scroll_spy = mock.MagicMock(wraps=tags_tab._scroll_results_to_top)  # type: ignore[attr-defined]
    tags_tab._scroll_results_to_top = scroll_spy  # type: ignore[attr-defined]

    tags_tab._on_load_more_clicked()
    _await_idle(tags_tab, qapp)

    scroll_spy.assert_not_called()

    tags_tab._query_edit.setText("fresh")  # type: ignore[attr-defined]
    tags_tab._on_search_clicked()
    _await_idle(tags_tab, qapp)

    scroll_spy.assert_called_once_with()


def test_async_search_cancel(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._query_edit.setText("again")  # type: ignore[attr-defined]
    tags_tab._on_search_clicked()
    assert _wait_for(lambda: tags_tab._search_busy, qapp, timeout=1.0)
    tags_tab._cancel_active_search()
    _await_idle(tags_tab, qapp)
    assert tags_tab._last_search_cancelled


def test_bootstrap_shows_latest_page(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    assert tags_tab._table_model.rowCount() == tags_tab._search_chunk_size  # type: ignore[attr-defined]
    assert tags_tab._offset == tags_tab._search_chunk_size  # type: ignore[attr-defined]
    expected = f"Showing {tags_tab._search_chunk_size} result(s) for '*'"  # type: ignore[attr-defined]
    assert tags_tab._status_label.text() == expected  # type: ignore[attr-defined]
    assert not tags_tab._search_busy  # type: ignore[attr-defined]
    assert tags_tab._search_worker is None  # type: ignore[attr-defined]


def test_typing_does_not_trigger_search(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    initial_rows = tags_tab._table_model.rowCount()  # type: ignore[attr-defined]
    tags_tab._completion_candidates = [TagMeta(name="tag0", category=0)]  # type: ignore[attr-defined]
    tags_tab._query_edit.setText("tag")  # type: ignore[attr-defined]
    tags_tab._on_query_text_edited("tag")

    assert _wait_for(lambda: tags_tab._search_worker is None, qapp, timeout=0.5)  # type: ignore[attr-defined]
    assert not tags_tab._search_busy  # type: ignore[attr-defined]
    assert tags_tab._table_model.rowCount() == initial_rows  # type: ignore[attr-defined]


def test_negative_autocomplete_displays_hyphen(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._completion_candidates = [TagMeta(name="tag0", category=0, count=1234)]  # type: ignore[attr-defined]
    tags_tab._query_edit.setText("-ta")  # type: ignore[attr-defined]
    tags_tab._query_edit.setCursorPosition(3)  # type: ignore[attr-defined]
    tags_tab._refresh_completions()  # type: ignore[attr-defined]

    index = tags_tab._tag_model.index(0, 0)  # type: ignore[attr-defined]
    assert index.data(Qt.ItemDataRole.DisplayRole) == "-tag0 (1.23k)"
    assert index.data(int(tags_tab._tag_model.NAME_ROLE)) == "tag0"  # type: ignore[attr-defined]


def test_logical_operators_are_not_autocomplete_candidates(tags_tab: TagsTab, qapp: QApplication) -> None:
    _await_idle(tags_tab, qapp)
    names = {tag.name for tag in tags_tab._completion_candidates}  # type: ignore[attr-defined]
    assert {"AND", "OR", "NOT"}.isdisjoint(names)

    tags_tab._query_edit.setText("-an")  # type: ignore[attr-defined]
    tags_tab._query_edit.setCursorPosition(3)  # type: ignore[attr-defined]
    tags_tab._refresh_completions()  # type: ignore[attr-defined]

    assert tags_tab._tag_model.rowCount() == 0  # type: ignore[attr-defined]


def test_delete_selected_result_trashes_file_and_marks_db_absent(
    tags_tab: TagsTab,
    qapp: QApplication,
    populated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _await_idle(tags_tab, qapp)
    trashed: list[str] = []

    monkeypatch.setattr("ui.tags_tab.trash_path", lambda path: trashed.append(str(path)))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    first_record = dict(tags_tab._results_cache[0])  # type: ignore[attr-defined]
    file_id = int(first_record["id"])
    path = str(first_record["path"])
    tags_tab._table_view.selectRow(0)  # type: ignore[attr-defined]
    tags_tab._on_delete_selected_result()  # type: ignore[attr-defined]

    assert _wait_for(lambda: not tags_tab._delete_active, qapp, timeout=3.0)  # type: ignore[attr-defined]
    assert trashed == [str(Path(path))]
    assert all(int(row["id"]) != file_id for row in tags_tab._results_cache)  # type: ignore[attr-defined]
    assert tags_tab._table_model.rowCount() == tags_tab._search_chunk_size - 1  # type: ignore[attr-defined]

    with sqlite3.connect(populated_db) as conn:
        row = conn.execute("SELECT is_present, deleted_at FROM files WHERE id = ?", (file_id,)).fetchone()
    assert row is not None
    assert row[0] == 0
    assert row[1] is not None


def test_delete_multiple_selected_results_trashes_files_and_marks_db_absent(
    tags_tab: TagsTab,
    qapp: QApplication,
    populated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._on_load_more_clicked()  # type: ignore[attr-defined]
    _await_idle(tags_tab, qapp)
    assert tags_tab._table_model.rowCount() >= 2  # type: ignore[attr-defined]
    trashed: list[str] = []

    monkeypatch.setattr("ui.tags_tab.trash_path", lambda path: trashed.append(str(path)))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    selected_records = [dict(tags_tab._results_cache[row]) for row in (0, 1)]  # type: ignore[attr-defined]
    file_ids = {int(record["id"]) for record in selected_records}
    expected_paths = [str(Path(str(record["path"]))) for record in selected_records]
    selection_model = tags_tab._table_view.selectionModel()  # type: ignore[attr-defined]
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    selection_model.select(tags_tab._table_model.index(0, 0), flags)  # type: ignore[attr-defined]
    selection_model.select(tags_tab._table_model.index(1, 0), flags)  # type: ignore[attr-defined]
    tags_tab._on_delete_selected_result()  # type: ignore[attr-defined]

    assert _wait_for(lambda: not tags_tab._delete_active, qapp, timeout=3.0)  # type: ignore[attr-defined]
    assert trashed == expected_paths
    assert file_ids.isdisjoint({int(row["id"]) for row in tags_tab._results_cache})  # type: ignore[attr-defined]

    with sqlite3.connect(populated_db) as conn:
        rows = conn.execute(
            f"SELECT id, is_present, deleted_at FROM files WHERE id IN ({', '.join('?' for _ in file_ids)})",
            tuple(file_ids),
        ).fetchall()
    assert len(rows) == 2
    assert all(row[1] == 0 and row[2] is not None for row in rows)


def test_delete_selected_result_does_not_mark_absent_when_trash_fails(
    tags_tab: TagsTab,
    qapp: QApplication,
    populated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _await_idle(tags_tab, qapp)
    warnings: list[str] = []

    def fail_trash(path: Path) -> None:
        raise RuntimeError("trash unavailable")

    monkeypatch.setattr("ui.tags_tab.trash_path", fail_trash)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2])))

    first_record = dict(tags_tab._results_cache[0])  # type: ignore[attr-defined]
    file_id = int(first_record["id"])
    tags_tab._table_view.selectRow(0)  # type: ignore[attr-defined]
    tags_tab._on_delete_selected_result()  # type: ignore[attr-defined]

    assert _wait_for(lambda: not tags_tab._delete_active, qapp, timeout=3.0)  # type: ignore[attr-defined]
    assert any(int(row["id"]) == file_id for row in tags_tab._results_cache)  # type: ignore[attr-defined]
    assert "Delete failed." in tags_tab._status_label.text()  # type: ignore[attr-defined]
    assert warnings and "trash unavailable" in warnings[0]
    with sqlite3.connect(populated_db) as conn:
        row = conn.execute("SELECT is_present, deleted_at FROM files WHERE id = ?", (file_id,)).fetchone()
    assert row == (1, None)


def test_delete_removes_trashed_result_when_db_update_fails(
    tags_tab: TagsTab,
    qapp: QApplication,
    populated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _await_idle(tags_tab, qapp)
    trashed: list[str] = []
    warnings: list[str] = []

    monkeypatch.setattr("ui.tags_tab.trash_path", lambda path: trashed.append(str(path)))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(str(args[2])))
    monkeypatch.setattr(
        tags_tab._view_model,  # type: ignore[attr-defined]
        "mark_files_absent",
        lambda conn, ids: (_ for _ in ()).throw(sqlite3.OperationalError("database locked")),
    )

    first_record = dict(tags_tab._results_cache[0])  # type: ignore[attr-defined]
    file_id = int(first_record["id"])
    tags_tab._table_view.selectRow(0)  # type: ignore[attr-defined]
    offset_before = tags_tab._offset  # type: ignore[attr-defined]
    tags_tab._on_delete_selected_result()  # type: ignore[attr-defined]

    assert _wait_for(lambda: not tags_tab._delete_active, qapp, timeout=3.0)  # type: ignore[attr-defined]
    assert trashed == [str(Path(str(first_record["path"])))]
    assert all(int(row["id"]) != file_id for row in tags_tab._results_cache)  # type: ignore[attr-defined]
    assert tags_tab._offset == offset_before  # type: ignore[attr-defined]
    assert warnings and "moved to trash, but DB update failed" in warnings[0]
    with sqlite3.connect(populated_db) as conn:
        row = conn.execute("SELECT is_present, deleted_at FROM files WHERE id = ?", (file_id,)).fetchone()
    assert row == (1, None)


def test_delete_grid_selection_resyncs_grid_row_roles(
    tags_tab: TagsTab,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _await_idle(tags_tab, qapp)
    tags_tab._on_load_more_clicked()  # type: ignore[attr-defined]
    _await_idle(tags_tab, qapp)
    assert tags_tab._grid_model.rowCount() >= 2  # type: ignore[attr-defined]

    monkeypatch.setattr("ui.tags_tab.trash_path", lambda path: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    tags_tab._grid_button.setChecked(True)  # type: ignore[attr-defined]
    selection_model = tags_tab._grid_view.selectionModel()  # type: ignore[attr-defined]
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    selection_model.select(tags_tab._grid_model.index(0, 0), flags)  # type: ignore[attr-defined]
    selection_model.select(tags_tab._grid_model.index(1, 0), flags)  # type: ignore[attr-defined]
    tags_tab._on_delete_selected_result()  # type: ignore[attr-defined]

    assert _wait_for(lambda: not tags_tab._delete_active, qapp, timeout=3.0)  # type: ignore[attr-defined]
    roles = [
        tags_tab._grid_model.item(row).data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined,union-attr]
        for row in range(tags_tab._grid_model.rowCount())  # type: ignore[attr-defined]
    ]
    assert roles == list(range(tags_tab._grid_model.rowCount()))  # type: ignore[attr-defined]
