"""Pure tree and thumbnail-state helpers for duplicate results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from dup.scanner import DuplicateCluster, DuplicateClusterEntry

T = TypeVar("T")


def bound_path_from_bindings(bindings: Mapping[str, Sequence[T]], target: T) -> Path | None:
    """Return the bound path for *target* from thumbnail binding lists."""

    for key, items in bindings.items():
        if target in items:
            try:
                return Path(key)
            except (TypeError, ValueError, OSError):
                return None
    return None


def should_start_thumbnail(key: str, *, inflight: set[str], done: set[str]) -> bool:
    """Return whether a thumbnail request should be started."""

    return key not in inflight and key not in done


def unique_pending_thumbnail_keys(keys: Iterable[str], *, inflight: set[str], done: set[str]) -> list[str]:
    """Return de-duplicated thumbnail keys that still need work."""

    pending: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen or not should_start_thumbnail(key, inflight=inflight, done=done):
            continue
        seen.add(key)
        pending.append(key)
    return pending


def cluster_hamming_score(cluster: DuplicateCluster) -> int:
    """Return the maximum non-keeper hamming score, or -1 when unavailable."""

    scores = [
        entry.best_hamming
        for entry in cluster.files
        if entry.file.file_id != cluster.keeper_id and entry.best_hamming is not None
    ]
    return max(scores) if scores else -1


def default_checked_entries(cluster: DuplicateCluster) -> list[DuplicateClusterEntry]:
    """Return entries treated as checked before a duplicate group is expanded."""

    return [entry for entry in cluster.files if entry.file.file_id != cluster.keeper_id]


__all__ = [
    "bound_path_from_bindings",
    "cluster_hamming_score",
    "default_checked_entries",
    "should_start_thumbnail",
    "unique_pending_thumbnail_keys",
]
