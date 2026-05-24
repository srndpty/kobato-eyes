"""Tests for duplicate tree state helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from dup.scanner import DuplicateCluster, DuplicateClusterEntry, DuplicateFile
from ui.dup_tree_state import (
    bound_path_from_bindings,
    cluster_hamming_score,
    default_checked_entries,
    should_start_thumbnail,
    unique_pending_thumbnail_keys,
)


def _entry(file_id: int, name: str, *, best_hamming: int | None = None) -> DuplicateClusterEntry:
    return DuplicateClusterEntry(
        file=DuplicateFile(
            file_id=file_id,
            path=Path(name),
            size=100,
            width=10,
            height=10,
            phash=file_id,
        ),
        best_hamming=best_hamming,
    )


def test_bound_path_from_bindings_returns_matching_path() -> None:
    target = object()

    assert bound_path_from_bindings({"C:/a.png": [target]}, target) == Path("C:/a.png")
    assert bound_path_from_bindings({"C:/a.png": []}, target) is None
    assert bound_path_from_bindings(cast(Any, {None: [target]}), target) is None


def test_thumbnail_queue_helpers_skip_inflight_done_and_duplicates() -> None:
    inflight = {"a.png"}
    done = {"b.png"}

    assert not should_start_thumbnail("a.png", inflight=inflight, done=done)
    assert not should_start_thumbnail("b.png", inflight=inflight, done=done)
    assert should_start_thumbnail("c.png", inflight=inflight, done=done)
    assert unique_pending_thumbnail_keys(["a.png", "c.png", "c.png", "d.png"], inflight=inflight, done=done) == [
        "c.png",
        "d.png",
    ]


def test_cluster_hamming_score_and_default_checked_entries_skip_keeper() -> None:
    keeper = _entry(1, "keeper.png", best_hamming=None)
    weak = _entry(2, "weak.png", best_hamming=4)
    strong = _entry(3, "strong.png", best_hamming=9)
    cluster = DuplicateCluster(files=[keeper, weak, strong], keeper_id=1)

    assert cluster_hamming_score(cluster) == 9
    assert default_checked_entries(cluster) == [weak, strong]
    assert cluster_hamming_score(DuplicateCluster(files=[keeper], keeper_id=1)) == -1
