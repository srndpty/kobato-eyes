"""Formatting helpers for index and refresh UI feedback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


@dataclass(frozen=True)
class IndexFeedback:
    """Status-bar and toast text produced after an indexing task."""

    status: str
    toast: str


def _as_int(value: object, default: int = 0) -> int:
    """Convert loose stats values to int for display."""

    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    """Convert loose stats values to float for display."""

    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def format_refresh_feedback(stats: Mapping[str, object], folders: Sequence[Path]) -> IndexFeedback:
    """Return status and toast text for a completed manual refresh."""

    elapsed = _as_float(stats.get("elapsed_sec"))
    queued = _as_int(stats.get("queued"))
    tagged = _as_int(stats.get("tagged"))
    missing = _as_int(stats.get("missing"))
    soft_deleted = _as_int(stats.get("soft_deleted"))
    hard_deleted = _as_int(stats.get("hard_deleted"))
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
    message = f"Indexed: {_as_int(stats.get('scanned'))} files / Tagged: {_as_int(stats.get('tagged'))}"
    retagged = _as_int(stats.get("retagged"))
    requested = _as_int(stats.get("retagged_marked"), retagged)
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
