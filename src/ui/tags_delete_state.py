"""Pure delete-state helpers for tag search results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def format_delete_confirmation(paths: Sequence[Path]) -> str:
    """Return confirmation text for deleting selected search results."""

    if len(paths) == 1:
        return f"Move this image to the trash and remove it from search results?\n\n{paths[0]}"
    preview = "\n".join(str(path) for path in paths[:5])
    suffix = "" if len(paths) <= 5 else f"\n... and {len(paths) - 5} more"
    return f"Move {len(paths)} images to the trash and remove them from search results?\n\n{preview}{suffix}"


def format_deleting_status(paths: Sequence[Path]) -> str:
    """Return status text for an active delete operation."""

    if len(paths) == 1:
        return f"Deleting {paths[0].name}…"
    return f"Deleting {len(paths)} images…"


def format_delete_result_status(
    removed: Sequence[tuple[int, str]],
    failures: Sequence[tuple[str, int, str, str]],
    total: int,
    remaining: int,
    query_label: str,
) -> str:
    """Return status text after a delete operation finishes."""

    if not removed:
        return f"Delete failed. Showing {remaining} result(s) for '{query_label}'"
    if len(removed) == 1 and total == 1:
        try:
            name = Path(removed[0][1]).name
        except IndexError:
            name = "image"
        return f"Deleted {name}. Showing {remaining} result(s) for '{query_label}'"
    if failures:
        return f"Deleted {len(removed)}/{total} image(s). Showing {remaining} result(s) for '{query_label}'"
    return f"Deleted {len(removed)} image(s). Showing {remaining} result(s) for '{query_label}'"


def format_delete_failure_reason(kind: str, reason: str) -> str:
    """Return a user-facing reason for a delete failure kind."""

    if kind == "db":
        return f"moved to trash, but DB update failed: {reason}"
    return reason


__all__ = [
    "format_delete_confirmation",
    "format_delete_failure_reason",
    "format_delete_result_status",
    "format_deleting_status",
]
