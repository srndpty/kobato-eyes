"""Formatting helpers for index and refresh UI feedback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class IndexFeedback:
    """Status-bar and toast text produced after an indexing task."""

    status: str
    toast: str


def format_refresh_feedback(stats: Mapping[str, object], folders: Sequence[Path]) -> IndexFeedback:
    """Return status and toast text for a completed manual refresh."""

    elapsed = float(stats.get("elapsed_sec", 0.0) or 0.0)
    queued = int(stats.get("queued", 0) or 0)
    tagged = int(stats.get("tagged", 0) or 0)
    missing = int(stats.get("missing", 0) or 0)
    soft_deleted = int(stats.get("soft_deleted", 0) or 0)
    hard_deleted = int(stats.get("hard_deleted", 0) or 0)
    removed_total = soft_deleted + hard_deleted
    if missing <= 0 and removed_total > 0:
        missing = removed_total
    hard_delete = bool(stats.get("hard_delete", False) or hard_deleted)
    removal_label = "hard delete" if hard_delete else "soft delete"
    folder_text = ", ".join(str(item) for item in folders) if folders else ""
    status = (
        f"Refresh complete: {tagged} tagged, {removed_total} missing removed "
        f"({removal_label}, {elapsed:.2f}s; queued {queued})."
    )
    if folder_text:
        status += f" [{folder_text}]"
    toast = f"{tagged} tagged; {removed_total} missing removed"
    if hard_delete:
        toast += " (hard delete)"
    if folder_text:
        toast = f"{toast} {folder_text}"
    return IndexFeedback(status=status, toast=toast)


def format_index_success_toast(stats: Mapping[str, object], *, retag_active: bool) -> str:
    """Return toast text for a completed index or retag operation."""

    tagger_name = str(stats.get("tagger_name") or "unknown")
    message = f"Indexed: {int(stats.get('scanned', 0))} files / Tagged: {int(stats.get('tagged', 0))}"
    retagged = int(stats.get("retagged", 0) or 0)
    requested = int(stats.get("retagged_marked", retagged) or 0)
    if retag_active:
        if requested and requested != retagged:
            message += f" / Retagged: {retagged}/{requested}"
        else:
            message += f" / Retagged: {retagged}"
    elif retagged:
        message += f" / Retagged: {retagged}"
    message += f" (tagger: {tagger_name})"
    return message


def format_index_failure(message: str, *, prefix: str, db_display: str, passthrough_message: str | None = None) -> str:
    """Return user-facing text for a failed indexing task."""

    if passthrough_message is not None and message == passthrough_message:
        return message
    return f"{prefix} failed (DB: {db_display}): {message}"


__all__ = ["IndexFeedback", "format_index_failure", "format_index_success_toast", "format_refresh_feedback"]
