"""Smoke tests for duplicate detection UI behaviour."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)

from PyQt6.QtWidgets import QApplication, QMessageBox

from dup.scanner import DuplicateCluster, DuplicateClusterEntry, DuplicateFile
from ui.dup_tab import DupTab
from ui.viewmodels import DupViewModel

pytestmark = pytest.mark.gui


class _Connection:
    def close(self) -> None:
        """Close the fake connection."""


@pytest.fixture(scope="module")
def qapp() -> Iterable[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def dup_tab(qapp: QApplication, tmp_path: Path) -> Iterable[DupTab]:
    view_model = DupViewModel(
        db_path=tmp_path / "library.db",
        connection_factory=lambda path: _Connection(),
        mark_files_absent=lambda conn, ids: None,
    )
    widget = DupTab(view_model=view_model)
    widget.show()
    qapp.processEvents()
    try:
        yield widget
    finally:
        widget.deleteLater()


def _entry(file_id: int, path: Path, *, size: int = 100) -> DuplicateClusterEntry:
    file = DuplicateFile(file_id=file_id, path=path, size=size, width=10, height=10, phash=file_id)
    return DuplicateClusterEntry(file=file, best_hamming=file_id)


def _cluster(tmp_path: Path) -> DuplicateCluster:
    entries = [
        _entry(1, tmp_path / "keep.png", size=200),
        _entry(2, tmp_path / "trash.png", size=100),
    ]
    return DuplicateCluster(files=entries, keeper_id=1)


def test_initial_action_state(dup_tab: DupTab) -> None:
    assert dup_tab._scan_button.isEnabled()  # type: ignore[attr-defined]
    assert not dup_tab._mark_button.isEnabled()  # type: ignore[attr-defined]
    assert not dup_tab._trash_button.isEnabled()  # type: ignore[attr-defined]
    assert not dup_tab._export_button.isEnabled()  # type: ignore[attr-defined]


def test_scan_start_disables_scan_button(monkeypatch, dup_tab: DupTab) -> None:
    submitted: list[object] = []

    class _Signals:
        def __init__(self) -> None:
            self.progressState = SimpleSignal()
            self.completed = SimpleSignal()
            self.error = SimpleSignal()

    class SimpleSignal:
        def connect(self, callback) -> None:
            submitted.append(callback)

    class _Handle:
        def __init__(self) -> None:
            self.signals = _Signals()

    class _Jobs:
        def submit_handle(self, job, priority):
            submitted.append((job, priority))
            return _Handle()

    dup_tab._jobs = _Jobs()  # type: ignore[attr-defined]

    dup_tab._on_scan_clicked()  # type: ignore[attr-defined]

    assert submitted
    assert not dup_tab._scan_button.isEnabled()  # type: ignore[attr-defined]
    assert dup_tab._status_label.text() == "Scanning duplicates..."  # type: ignore[attr-defined]


def test_scan_error_restores_button(monkeypatch, dup_tab: DupTab) -> None:
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: None)
    dup_tab._scan_button.setEnabled(False)  # type: ignore[attr-defined]

    dup_tab._on_scan_error("boom")  # type: ignore[attr-defined]

    assert dup_tab._scan_button.isEnabled()  # type: ignore[attr-defined]
    assert dup_tab._status_label.text() == "Duplicate scan failed."  # type: ignore[attr-defined]


def test_cluster_enables_actions_and_trash_updates_state(monkeypatch, dup_tab: DupTab, tmp_path: Path) -> None:
    cluster = _cluster(tmp_path)
    marked: list[list[int]] = []
    dup_tab._clusters = [cluster]  # type: ignore[attr-defined]
    dup_tab._populate_tree()  # type: ignore[attr-defined]
    dup_tab._populate_tick()  # type: ignore[attr-defined]
    dup_tab._update_action_states()  # type: ignore[attr-defined]

    assert dup_tab._mark_button.isEnabled()  # type: ignore[attr-defined]
    assert dup_tab._trash_button.isEnabled()  # type: ignore[attr-defined]

    dup_tab._view_model.mark_files_absent = lambda conn, ids: marked.append(list(ids))  # type: ignore[attr-defined,assignment]
    monkeypatch.setattr("ui.dup_actions.trash_duplicate_entries", lambda entries: (list(entries), []))

    dup_tab._on_trash_checked()  # type: ignore[attr-defined]

    assert marked == [[2]]
    assert dup_tab._status_label.text() == "Moved 1 file(s) to trash."  # type: ignore[attr-defined]


def test_mark_and_uncheck_actions_update_thumb_panel_tiles(dup_tab: DupTab, tmp_path: Path) -> None:
    cluster = _cluster(tmp_path)
    dup_tab._clusters = [cluster]  # type: ignore[attr-defined]
    dup_tab._populate_tree()  # type: ignore[attr-defined]
    dup_tab._populate_tick()  # type: ignore[attr-defined]
    top = dup_tab._tree.topLevelItem(0)  # type: ignore[attr-defined]
    assert top is not None

    dup_tab._build_children_for_cluster(top, cluster)  # type: ignore[attr-defined]
    panel = dup_tab._panel_of_group(top)  # type: ignore[attr-defined]
    assert panel is not None

    for tile in panel.tiles:
        tile.set_checked(False)

    dup_tab._on_mark_keep_largest()  # type: ignore[attr-defined]

    checked_ids = {entry.file.file_id for entry in dup_tab._iter_checked_entries()}  # type: ignore[attr-defined]
    assert checked_ids == {2}
    assert {tile.entry.file.file_id: tile.is_checked() for tile in panel.tiles} == {1: False, 2: True}

    dup_tab._on_uncheck_all()  # type: ignore[attr-defined]

    assert list(dup_tab._iter_checked_entries()) == []  # type: ignore[attr-defined]
    assert all(not tile.is_checked() for tile in panel.tiles)


def test_duplicate_thumbnail_queue_deduplicates_pending_work(dup_tab: DupTab, tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"

    dup_tab._queue_thumb(image_path)  # type: ignore[attr-defined]
    dup_tab._queue_thumb(image_path)  # type: ignore[attr-defined]

    assert list(dup_tab._thumb_pending).count(str(image_path)) == 1  # type: ignore[attr-defined]

    dup_tab._thumb_inflight.add(str(image_path))  # type: ignore[attr-defined]
    dup_tab._queue_thumb(image_path)  # type: ignore[attr-defined]

    assert list(dup_tab._thumb_pending).count(str(image_path)) == 1  # type: ignore[attr-defined]
