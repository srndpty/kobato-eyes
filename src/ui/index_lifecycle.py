"""Pure lifecycle helpers for indexing and refresh UI state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

from ui.index_feedback import format_index_failure, format_index_success_toast, format_refresh_feedback

IndexMode = Literal["refresh", "retag", "index"]
ConnectionRetryAction = Literal["retry", "give_up", "raise"]


@dataclass(frozen=True)
class IndexFinishPlan:
    """UI effects to apply after an index task finishes."""

    status: str
    toast: str
    refresh_active: bool
    retag_active: bool
    active_refresh_folder: Sequence[Path] | None
    run_search: bool


@dataclass(frozen=True)
class IndexFailurePlan:
    """UI effects to apply after an index task fails."""

    status: str
    toast: str
    refresh_active: bool = False
    retag_active: bool = False


def index_mode(*, refresh_active: bool, retag_active: bool) -> IndexMode:
    """Return the active index mode from UI flags."""

    if refresh_active:
        return "refresh"
    if retag_active:
        return "retag"
    return "index"


def index_prefix(mode: IndexMode) -> str:
    """Return the user-facing activity prefix for an index mode."""

    return {"refresh": "Refreshing", "retag": "Retagging", "index": "Indexing"}[mode]


def index_started_status(*, refresh_active: bool, retag_active: bool) -> str:
    """Return status text for a newly-started index task."""

    return f"{index_prefix(index_mode(refresh_active=refresh_active, retag_active=retag_active))}…"


def index_cancel_status(*, refresh_active: bool, retag_active: bool) -> str:
    """Return status text after the user requested cancellation."""

    return f"{index_prefix(index_mode(refresh_active=refresh_active, retag_active=retag_active))} cancelling…"


def plan_index_finished(
    stats: Mapping[str, object],
    *,
    refresh_active: bool,
    retag_active: bool,
    active_refresh_folder: Sequence[Path] | None,
    has_current_query: bool,
) -> IndexFinishPlan:
    """Build UI state changes for a finished index task."""

    elapsed = _as_float(stats.get("elapsed_sec"))
    cancelled = bool(stats.get("cancelled", False))
    mode = index_mode(refresh_active=refresh_active, retag_active=retag_active)
    prefix = index_prefix(mode)

    if mode == "refresh":
        if cancelled:
            return IndexFinishPlan(
                status=f"Refresh cancelled after {elapsed:.2f}s.",
                toast="Refresh cancelled.",
                refresh_active=False,
                retag_active=retag_active,
                active_refresh_folder=None,
                run_search=False,
            )
        folders = list(active_refresh_folder or _folders_from_stats(stats))
        feedback = format_refresh_feedback(stats, folders)
        return IndexFinishPlan(
            status=feedback.status,
            toast=feedback.toast,
            refresh_active=False,
            retag_active=retag_active,
            active_refresh_folder=None,
            run_search=has_current_query,
        )

    if cancelled:
        return IndexFinishPlan(
            status=f"{prefix} cancelled after {elapsed:.2f}s.",
            toast=f"{prefix} cancelled.",
            refresh_active=refresh_active,
            retag_active=False,
            active_refresh_folder=active_refresh_folder,
            run_search=False,
        )

    status = f"{prefix} complete in {elapsed:.2f}s."
    return IndexFinishPlan(
        status=status,
        toast=format_index_success_toast(stats, retag_active=retag_active),
        refresh_active=refresh_active,
        retag_active=False,
        active_refresh_folder=active_refresh_folder,
        run_search=True,
    )


def plan_index_failed(
    message: str,
    *,
    refresh_active: bool,
    retag_active: bool,
    db_display: str,
    passthrough_message: str | None = None,
) -> IndexFailurePlan:
    """Build UI state changes for a failed index task."""

    prefix = index_prefix(index_mode(refresh_active=refresh_active, retag_active=retag_active))
    status = format_index_failure(
        message,
        prefix=prefix,
        db_display=db_display,
        passthrough_message=passthrough_message,
    )
    return IndexFailurePlan(status=status, toast=status)


def connection_retry_action(exc: Exception, attempts: int) -> ConnectionRetryAction:
    """Return whether a connection restore error should be retried."""

    message = str(exc).lower()
    if "locked" not in message and "busy" not in message:
        return "raise"
    if attempts <= 1:
        return "give_up"
    return "retry"


def _folders_from_stats(stats: Mapping[str, object]) -> list[Path]:
    roots_meta = stats.get("roots", [])
    if not isinstance(roots_meta, Sequence) or isinstance(roots_meta, (str, bytes)):
        return []
    folders: list[Path] = []
    for entry in roots_meta:
        if not isinstance(entry, Mapping):
            continue
        folder = entry.get("folder")
        if folder:
            folders.append(Path(str(folder)))
    return folders


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


__all__ = [
    "ConnectionRetryAction",
    "IndexFailurePlan",
    "IndexFinishPlan",
    "IndexMode",
    "connection_retry_action",
    "index_cancel_status",
    "index_mode",
    "index_prefix",
    "index_started_status",
    "plan_index_failed",
    "plan_index_finished",
]
