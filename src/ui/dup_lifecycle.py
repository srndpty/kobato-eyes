"""Pure lifecycle helpers for duplicate detection UI state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, cast

from ui.dup_status import format_duplicate_scan_complete, format_duplicate_summary


class _ClusterLike(Protocol):
    files: Sequence[object]


@dataclass(frozen=True)
class DuplicateActionAvailability:
    """Enabled state for duplicate action buttons."""

    mark: bool
    uncheck: bool
    export: bool
    trash: bool


@dataclass(frozen=True)
class DuplicateProgressState:
    """Progress bar state for duplicate tasks."""

    maximum: int
    value: int


@dataclass(frozen=True)
class DuplicateScanPlan:
    """UI state derived from a duplicate scan result payload."""

    clusters: list[object]
    status: str
    refine: bool
    clear_tree: bool
    valid_payload: bool


@dataclass(frozen=True)
class RefineProgressState:
    """Progress dialog state for duplicate refinement."""

    label: str
    maximum: int
    value: int


def duplicate_action_availability(*, has_clusters: bool, checked_count: int) -> DuplicateActionAvailability:
    """Return enabled states for duplicate action buttons."""

    return DuplicateActionAvailability(
        mark=has_clusters,
        uncheck=has_clusters,
        export=has_clusters,
        trash=checked_count > 0,
    )


def duplicate_scan_progress(current: int, total: int) -> DuplicateProgressState:
    """Return progress bar state for duplicate scanning."""

    if total <= 0:
        return DuplicateProgressState(maximum=1, value=0)
    return DuplicateProgressState(maximum=total, value=current)


def duplicate_scan_finished_plan(payload: object, cluster_type: type[object]) -> DuplicateScanPlan:
    """Return UI state for a duplicate scan result."""

    if not isinstance(payload, list):
        return DuplicateScanPlan(
            clusters=[],
            status="Scan completed with unexpected payload",
            refine=False,
            clear_tree=False,
            valid_payload=False,
        )
    clusters = [cluster for cluster in payload if isinstance(cluster, cluster_type)]
    cluster_likes = cast(Sequence[_ClusterLike], clusters)
    if not clusters:
        return DuplicateScanPlan(
            clusters=[],
            status="No duplicate groups detected.",
            refine=False,
            clear_tree=True,
            valid_payload=True,
        )
    return DuplicateScanPlan(
        clusters=clusters,
        status=format_duplicate_scan_complete(cluster_likes),
        refine=True,
        clear_tree=False,
        valid_payload=True,
    )


def duplicate_refine_progress(current: int, total: int, stage: str) -> RefineProgressState:
    """Return progress dialog state for duplicate refinement."""

    maximum = total if total > 0 else 1
    return RefineProgressState(
        label=f"{stage}  {current} / {maximum}",
        maximum=maximum,
        value=current,
    )


def duplicate_refine_complete_status(clusters: Sequence[_ClusterLike]) -> str:
    """Return status text after duplicate refinement completes."""

    return f"Refine complete: {format_duplicate_summary(clusters)}"


def duplicate_refine_cancel_status() -> str:
    """Return status text after duplicate refinement is cancelled."""

    return "Refine canceled."


def duplicate_refine_error_status() -> str:
    """Return status text after duplicate refinement fails."""

    return "Refine failed."


def duplicate_trash_summary(success_count: int, failure_count: int) -> str:
    """Return status text after trashing duplicate entries."""

    summary = f"Moved {success_count} file(s) to trash."
    if failure_count:
        summary += f" Failed: {failure_count}."
    return summary


def duplicate_export_status(file_path: str) -> str:
    """Return status text after duplicate CSV export."""

    return f"Exported duplicate groups to {file_path}."


__all__ = [
    "DuplicateActionAvailability",
    "DuplicateProgressState",
    "DuplicateScanPlan",
    "RefineProgressState",
    "duplicate_action_availability",
    "duplicate_export_status",
    "duplicate_refine_cancel_status",
    "duplicate_refine_complete_status",
    "duplicate_refine_error_status",
    "duplicate_refine_progress",
    "duplicate_scan_finished_plan",
    "duplicate_scan_progress",
    "duplicate_trash_summary",
]
