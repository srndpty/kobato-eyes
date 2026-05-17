"""Tests for OS-independent file action helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from dup.scanner import DuplicateCluster, DuplicateClusterEntry, DuplicateFile
from ui import file_actions


def _entry(file_id: int, path: Path, *, size: int = 100, hamming: int | None = None) -> DuplicateClusterEntry:
    file = DuplicateFile(file_id=file_id, path=path, size=size, width=10, height=20, phash=file_id)
    return DuplicateClusterEntry(file=file, best_hamming=hamming)


def test_export_duplicate_clusters_csv_writes_keeper_and_hamming(tmp_path: Path) -> None:
    export_path = tmp_path / "duplicates.csv"
    cluster = DuplicateCluster(
        files=[
            _entry(1, tmp_path / "keep.png", size=200, hamming=None),
            _entry(2, tmp_path / "trash.png", size=100, hamming=4),
        ],
        keeper_id=1,
    )

    file_actions.export_duplicate_clusters_csv([cluster], export_path)

    with export_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ["group", "file_id", "path", "size", "width", "height", "keeper", "hamming"],
        ["1", "1", (tmp_path / "keep.png").as_posix(), "200", "10", "20", "1", ""],
        ["1", "2", (tmp_path / "trash.png").as_posix(), "100", "10", "20", "0", "4"],
    ]


def test_trash_duplicate_entries_collects_successes_and_failures(monkeypatch, tmp_path: Path) -> None:
    success = _entry(1, tmp_path / "success.png")
    failure = _entry(2, tmp_path / "failure.png")
    trashed: list[str] = []

    def fake_send2trash(path: str) -> None:
        trashed.append(path)
        if path.endswith("failure.png"):
            raise OSError("trash unavailable")

    monkeypatch.setattr(file_actions, "send2trash", fake_send2trash)

    successes, failures = file_actions.trash_duplicate_entries([success, failure])

    assert successes == [success]
    assert failures == [(failure, "trash unavailable")]
    assert trashed == [str(success.file.path), str(failure.file.path)]
