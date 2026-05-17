"""Cluster update helpers for duplicate result actions."""

from __future__ import annotations

from collections.abc import Sequence

from dup.scanner import DuplicateCluster, DuplicateClusterEntry


def choose_keeper(entries: Sequence[DuplicateClusterEntry]) -> int:
    """Return the best keeper file ID among duplicate entries."""

    def key(entry: DuplicateClusterEntry) -> tuple[int, int, int, str, str, int]:
        file = entry.file
        return (
            -(file.size or 0),
            -file.resolution,
            -file.extension_priority,
            file.path.suffix.lower(),
            file.path.name.lower(),
            file.file_id,
        )

    return min(entries, key=key).file.file_id


def sort_entries_for_display(
    entries: Sequence[DuplicateClusterEntry],
    keeper_id: int,
) -> list[DuplicateClusterEntry]:
    """Return entries in the display order used by the duplicate UI."""

    return sorted(
        entries,
        key=lambda entry: (
            0 if entry.file.file_id == keeper_id else 1,
            -(entry.file.size or 0),
            -entry.file.resolution,
            -entry.file.extension_priority,
            entry.file.path.name.lower(),
            entry.file.file_id,
        ),
    )


def rebuild_cluster_after_removal(
    cluster: DuplicateCluster,
    removed_ids: set[int],
) -> DuplicateCluster | None:
    """Rebuild one cluster after files were removed, dropping singleton groups."""

    remaining = [entry for entry in cluster.files if entry.file.file_id not in removed_ids]
    if len(remaining) < 2:
        return None
    keeper_id = choose_keeper(remaining)
    return DuplicateCluster(files=sort_entries_for_display(remaining, keeper_id), keeper_id=keeper_id)


def rebuild_clusters_after_removal(
    clusters: Sequence[DuplicateCluster],
    removed_ids: set[int],
) -> list[DuplicateCluster]:
    """Rebuild duplicate clusters after trash/export actions changed file presence."""

    rebuilt: list[DuplicateCluster] = []
    for cluster in clusters:
        next_cluster = rebuild_cluster_after_removal(cluster, removed_ids)
        if next_cluster is not None:
            rebuilt.append(next_cluster)
    return rebuilt


__all__ = [
    "choose_keeper",
    "rebuild_cluster_after_removal",
    "rebuild_clusters_after_removal",
    "sort_entries_for_display",
]
