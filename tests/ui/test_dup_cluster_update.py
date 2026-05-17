"""Tests for duplicate cluster update helpers."""

from __future__ import annotations

from pathlib import Path

from dup.scanner import DuplicateCluster, DuplicateClusterEntry, DuplicateFile
from ui.dup_cluster_update import (
    choose_keeper,
    rebuild_cluster_after_removal,
    rebuild_clusters_after_removal,
    sort_entries_for_display,
)


def _entry(file_id: int, name: str, *, size: int, width: int = 10, height: int = 10) -> DuplicateClusterEntry:
    file = DuplicateFile(
        file_id=file_id,
        path=Path(name),
        size=size,
        width=width,
        height=height,
        phash=file_id,
    )
    return DuplicateClusterEntry(file=file, best_hamming=file_id)


def test_choose_keeper_prefers_larger_file_then_resolution() -> None:
    small = _entry(1, "small.png", size=100, width=100, height=100)
    large = _entry(2, "large.jpg", size=200, width=50, height=50)
    larger_resolution = _entry(3, "large-resolution.webp", size=200, width=200, height=200)

    assert choose_keeper([small, large, larger_resolution]) == 3


def test_sort_entries_for_display_keeps_keeper_first() -> None:
    keeper = _entry(1, "keeper.png", size=100)
    larger = _entry(2, "larger.png", size=300)
    smaller = _entry(3, "smaller.png", size=50)

    sorted_entries = sort_entries_for_display([smaller, larger, keeper], keeper_id=1)

    assert [entry.file.file_id for entry in sorted_entries] == [1, 2, 3]


def test_rebuild_cluster_after_removal_drops_singleton_groups() -> None:
    cluster = DuplicateCluster(
        files=[
            _entry(1, "keep.png", size=300),
            _entry(2, "trash-a.png", size=200),
            _entry(3, "trash-b.png", size=100),
        ],
        keeper_id=1,
    )

    rebuilt = rebuild_cluster_after_removal(cluster, {1})

    assert rebuilt is not None
    assert rebuilt.keeper_id == 2
    assert [entry.file.file_id for entry in rebuilt.files] == [2, 3]
    assert rebuild_cluster_after_removal(cluster, {1, 2}) is None


def test_rebuild_clusters_after_removal_filters_empty_results() -> None:
    first = DuplicateCluster(files=[_entry(1, "a.png", size=1), _entry(2, "b.png", size=2)], keeper_id=2)
    second = DuplicateCluster(files=[_entry(3, "c.png", size=3), _entry(4, "d.png", size=4)], keeper_id=4)

    rebuilt = rebuild_clusters_after_removal([first, second], {1, 3})

    assert rebuilt == []
