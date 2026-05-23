"""OS file actions used by UI tabs."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Sequence

from send2trash import send2trash

from dup.scanner import DuplicateCluster, DuplicateClusterEntry


def open_path(path: Path) -> None:
    """Open a file with the operating system default application."""

    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def reveal_in_file_manager(path: Path) -> None:
    """Reveal a file in the operating system file manager."""

    if os.name == "nt":
        subprocess.Popen(["explorer", "/select,", str(path)])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


def trash_path(path: Path) -> None:
    """Move a single filesystem path to the operating system trash."""

    send2trash(str(path))


def trash_duplicate_entries(
    entries: Sequence[DuplicateClusterEntry],
) -> tuple[list[DuplicateClusterEntry], list[tuple[DuplicateClusterEntry, str]]]:
    """Move duplicate entries to trash and collect per-file failures."""

    successes: list[DuplicateClusterEntry] = []
    failures: list[tuple[DuplicateClusterEntry, str]] = []
    for entry in entries:
        try:
            trash_path(entry.file.path)
            successes.append(entry)
        except Exception as exc:
            failures.append((entry, str(exc)))
    return successes, failures


def export_duplicate_clusters_csv(clusters: Sequence[DuplicateCluster], file_path: str | Path) -> None:
    """Export duplicate clusters to CSV."""

    with open(file_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "file_id", "path", "size", "width", "height", "keeper", "hamming"])
        for group_index, cluster in enumerate(clusters, start=1):
            for entry in cluster.files:
                writer.writerow(
                    [
                        group_index,
                        entry.file.file_id,
                        entry.file.path.as_posix(),
                        entry.file.size or 0,
                        entry.file.width or 0,
                        entry.file.height or 0,
                        1 if entry.file.file_id == cluster.keeper_id else 0,
                        entry.best_hamming if entry.best_hamming is not None else "",
                    ]
                )


__all__ = [
    "export_duplicate_clusters_csv",
    "open_path",
    "reveal_in_file_manager",
    "trash_duplicate_entries",
    "trash_path",
]
