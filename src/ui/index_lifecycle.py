"""Pure lifecycle helpers for indexing and refresh UI state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

from core.pipeline import IndexPhase, IndexProgress
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


@dataclass(frozen=True)
class IndexProgressState:
    """Progress dialog state derived from an indexing progress event."""

    label: str
    status: str
    title: str
    maximum: int
    value: int
    percent: int | None
    indeterminate: bool
    stage_index: int
    stage_total: int


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

    return f"{index_prefix(index_mode(refresh_active=refresh_active, retag_active=retag_active))}..."


def index_cancel_status(*, refresh_active: bool, retag_active: bool) -> str:
    """Return status text after the user requested cancellation."""

    return f"{index_prefix(index_mode(refresh_active=refresh_active, retag_active=retag_active))} cancelling..."


def index_phase_label(phase: IndexPhase) -> str:
    """Return a stable user-facing label for an index pipeline phase."""

    return {
        IndexPhase.SCAN: "Scanning",
        IndexPhase.PREPARE: "Preparing",
        IndexPhase.TAG: "Tagging",
        IndexPhase.FTS: "Writing index",
        IndexPhase.DONE: "Finishing",
    }[phase]


def index_phase_position(phase: IndexPhase) -> tuple[int, int]:
    """Return the stable ordinal position for an index pipeline phase."""

    phases = [
        IndexPhase.SCAN,
        IndexPhase.PREPARE,
        IndexPhase.TAG,
        IndexPhase.FTS,
        IndexPhase.DONE,
    ]
    return phases.index(phase) + 1, len(phases)


def index_progress_detail(progress: IndexProgress) -> str | None:
    """Return a user-facing detail for a progress event within its phase."""

    raw_message = str(progress.message or "").strip()
    if progress.phase is IndexPhase.FTS:
        return {
            "write": "Writing queued results",
            "merge.start": "Preparing DB merge",
            "merge.delete": "Replacing old tags",
            "merge.insert": "Writing tags",
            "merge.update": "Updating file metadata",
            "merge.index": "Rebuilding DB indexes",
            "merge.done": "Finalizing DB merge",
            "rebuild": "Rebuilding search index",
            "done": "Writing complete",
        }.get(raw_message, raw_message or None)
    if progress.total <= 0:
        return raw_message or None
    return None


def index_progress_state(progress: IndexProgress) -> IndexProgressState:
    """Return display-ready progress state for an indexing event."""

    phase_label = index_phase_label(progress.phase)
    stage_index, stage_total = index_phase_position(progress.phase)
    stage_label = f"{phase_label} ({stage_index}/{stage_total})"
    done = max(0, int(progress.done))
    total = int(progress.total)
    status = f"{stage_label}..."
    detail = index_progress_detail(progress)

    if total <= 0:
        label = detail if detail else "Working..."
        return IndexProgressState(
            label=label,
            status=status,
            title=stage_label,
            maximum=0,
            value=0,
            percent=None,
            indeterminate=True,
            stage_index=stage_index,
            stage_total=stage_total,
        )

    maximum = max(1, total)
    value = min(done, maximum)
    percent = min(100, (value * 100) // maximum)
    label_prefix = f"{detail}: " if detail else ""
    label = f"{label_prefix}{value} / {maximum} ({percent}%)"
    return IndexProgressState(
        label=label,
        status=status,
        title=stage_label,
        maximum=maximum,
        value=value,
        percent=percent,
        indeterminate=False,
        stage_index=stage_index,
        stage_total=stage_total,
    )


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

    write_failed = bool(stats.get("write_failed", False))
    write_error = str(stats.get("write_error", ""))
    status = f"{prefix} complete in {elapsed:.2f}s."
    if write_failed:
        status += f"  ※検索インデックス更新失敗（検索結果が古い可能性あり）: {write_error}"
    # write_failed でも run_search=True を維持する:
    # ファイルレコード自体は書き込み済みのため、既存インデックスで検索結果を更新する方が
    # 何も更新しないより UX として優れている（鮮度の問題はステータスメッセージで通知済み）
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
    "IndexProgressState",
    "connection_retry_action",
    "index_cancel_status",
    "index_mode",
    "index_phase_label",
    "index_phase_position",
    "index_progress_detail",
    "index_progress_state",
    "index_prefix",
    "index_started_status",
    "plan_index_failed",
    "plan_index_finished",
]
