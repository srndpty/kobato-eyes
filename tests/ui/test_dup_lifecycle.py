"""Tests for duplicate lifecycle helpers."""

from __future__ import annotations

from types import SimpleNamespace

from ui.dup_lifecycle import (
    duplicate_action_availability,
    duplicate_export_status,
    duplicate_refine_cancel_status,
    duplicate_refine_complete_status,
    duplicate_refine_error_status,
    duplicate_refine_progress,
    duplicate_scan_finished_plan,
    duplicate_scan_progress,
    duplicate_trash_summary,
)


class _Cluster:
    def __init__(self, files: list[object]) -> None:
        self.files = files


def test_duplicate_action_availability_follows_clusters_and_checked_count() -> None:
    assert duplicate_action_availability(has_clusters=False, checked_count=1).mark is False
    available = duplicate_action_availability(has_clusters=True, checked_count=2)
    assert available.mark is True
    assert available.uncheck is True
    assert available.export is True
    assert available.trash is True


def test_duplicate_scan_progress_handles_unknown_total() -> None:
    assert duplicate_scan_progress(5, 0).maximum == 1
    assert duplicate_scan_progress(5, 0).value == 0
    assert duplicate_scan_progress(3, 10).maximum == 10
    assert duplicate_scan_progress(3, 10).value == 3


def test_duplicate_scan_finished_plan_filters_payload_and_requests_refine() -> None:
    cluster = _Cluster([object(), object()])
    plan = duplicate_scan_finished_plan([cluster, SimpleNamespace(files=[])], _Cluster)

    assert plan.clusters == [cluster]
    assert plan.status == "Scan complete: 1 group(s), 2 file(s)."
    assert plan.refine is True
    assert plan.clear_tree is False
    assert plan.valid_payload is True


def test_duplicate_scan_finished_plan_handles_empty_and_invalid_payload() -> None:
    empty = duplicate_scan_finished_plan([], _Cluster)
    invalid = duplicate_scan_finished_plan(object(), _Cluster)

    assert empty.status == "No duplicate groups detected."
    assert empty.clear_tree is True
    assert empty.refine is False
    assert invalid.status == "Scan completed with unexpected payload"
    assert invalid.valid_payload is False


def test_duplicate_refine_progress_and_status_text() -> None:
    progress = duplicate_refine_progress(2, 0, "tile")

    assert progress.label == "tile  2 / 1"
    assert progress.maximum == 1
    assert progress.value == 2
    assert (
        duplicate_refine_complete_status([_Cluster([object()])]) == "Refine complete: 1 group(s), 1 file(s) detected."
    )
    assert duplicate_refine_cancel_status() == "Refine canceled."
    assert duplicate_refine_error_status() == "Refine failed."


def test_duplicate_trash_and_export_status_text() -> None:
    assert duplicate_trash_summary(2, 0) == "Moved 2 file(s) to trash."
    assert duplicate_trash_summary(1, 3) == "Moved 1 file(s) to trash. Failed: 3."
    assert duplicate_export_status("duplicates.csv") == "Exported duplicate groups to duplicates.csv."
