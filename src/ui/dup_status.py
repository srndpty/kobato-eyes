"""Small status helpers for duplicate detection UI."""

from __future__ import annotations

from typing import Protocol, Sequence


class _ClusterLike(Protocol):
    files: Sequence[object]


def duplicate_summary(clusters: Sequence[_ClusterLike]) -> tuple[int, int]:
    """Return the number of duplicate groups and files."""

    groups = len(clusters)
    files = sum(len(cluster.files) for cluster in clusters)
    return groups, files


def format_duplicate_summary(clusters: Sequence[_ClusterLike]) -> str:
    """Return status text for the current duplicate groups."""

    groups, files = duplicate_summary(clusters)
    return f"{groups} group(s), {files} file(s) detected."


def format_duplicate_scan_complete(clusters: Sequence[_ClusterLike]) -> str:
    """Return status text for a completed duplicate scan."""

    groups, files = duplicate_summary(clusters)
    return f"Scan complete: {groups} group(s), {files} file(s)."


__all__ = ["duplicate_summary", "format_duplicate_scan_complete", "format_duplicate_summary"]
