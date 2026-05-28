"""Action orchestration helpers for duplicate detection UI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

from dup.scanner import DuplicateCluster, DuplicateClusterEntry
from ui.dup_cluster_update import rebuild_clusters_after_removal
from ui.dup_lifecycle import duplicate_export_status, duplicate_trash_summary
from ui.file_actions import export_duplicate_clusters_csv, open_path, reveal_in_file_manager, trash_duplicate_entries


def trash_checked_duplicates(
    *,
    checked_entries: Sequence[DuplicateClusterEntry],
    clusters: Sequence[DuplicateCluster],
    db_path: Path,
    open_connection: Callable[[Path], object],
    mark_files_absent: Callable[[object, Sequence[int]], None],
) -> tuple[list[DuplicateCluster], str, list[tuple[DuplicateClusterEntry, str]]]:
    """Trash checked duplicate entries and return updated clusters, status, and failures."""

    successes, failures = trash_duplicate_entries(checked_entries)
    if successes:
        try:
            with closing(open_connection(db_path)) as conn:
                mark_files_absent(conn, [entry.file.file_id for entry in successes])
        except Exception as exc:  # pragma: no cover - database errors surfaced via UI
            failures.extend((entry, str(exc)) for entry in successes)
            successes.clear()
    removed_ids = {entry.file.file_id for entry in successes}
    updated_clusters = list(clusters)
    if removed_ids:
        updated_clusters = rebuild_clusters_after_removal(updated_clusters, removed_ids)
    return updated_clusters, duplicate_trash_summary(len(successes), len(failures)), failures


def export_duplicates_csv(clusters: Sequence[DuplicateCluster], file_path: str) -> str:
    """Export duplicate clusters to CSV and return the status message."""

    export_duplicate_clusters_csv(clusters, file_path)
    return duplicate_export_status(file_path)


def open_duplicate_path(path: Path) -> None:
    """Open a duplicate file path using the platform handler."""

    open_path(path)


def reveal_duplicate_path(path: Path) -> None:
    """Reveal a duplicate file path in the platform file manager."""

    reveal_in_file_manager(path)


__all__ = [
    "export_duplicates_csv",
    "open_duplicate_path",
    "reveal_duplicate_path",
    "trash_checked_duplicates",
]
