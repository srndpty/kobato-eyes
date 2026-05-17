"""Tests for index lifecycle helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tagger.wd14_onnx import ONNXRUNTIME_MISSING_MESSAGE
from ui.index_lifecycle import (
    connection_retry_action,
    index_cancel_status,
    index_started_status,
    plan_index_failed,
    plan_index_finished,
)


def test_index_started_and_cancel_status_follow_active_mode() -> None:
    assert index_started_status(refresh_active=True, retag_active=False) == "Refreshing…"
    assert index_started_status(refresh_active=False, retag_active=True) == "Retagging…"
    assert index_started_status(refresh_active=False, retag_active=False) == "Indexing…"
    assert index_cancel_status(refresh_active=False, retag_active=True) == "Retagging cancelling…"


def test_plan_index_finished_refresh_success_uses_active_folder(tmp_path: Path) -> None:
    plan = plan_index_finished(
        {"queued": 2, "tagged": 1, "missing": 0, "soft_deleted": 1, "elapsed_sec": 0.5},
        refresh_active=True,
        retag_active=False,
        active_refresh_folder=[tmp_path],
        has_current_query=True,
    )

    assert plan.status == f"Refresh complete: 1 tagged, 1 missing removed (soft delete, 0.50s; queued 2). [{tmp_path}]"
    assert plan.toast == f"1 tagged; 1 missing removed {tmp_path}"
    assert plan.refresh_active is False
    assert plan.run_search is True
    assert plan.active_refresh_folder is None


def test_plan_index_finished_cancelled_retag_resets_retag_without_search() -> None:
    plan = plan_index_finished(
        {"elapsed_sec": 3.25, "cancelled": True},
        refresh_active=False,
        retag_active=True,
        active_refresh_folder=None,
        has_current_query=True,
    )

    assert plan.status == "Retagging cancelled after 3.25s."
    assert plan.toast == "Retagging cancelled."
    assert plan.retag_active is False
    assert plan.run_search is False


def test_plan_index_finished_retag_success_searches_and_clears_retag() -> None:
    plan = plan_index_finished(
        {"elapsed_sec": 1.0, "scanned": 5, "tagged": 3, "retagged": 2, "tagger_name": "wd14"},
        refresh_active=False,
        retag_active=True,
        active_refresh_folder=None,
        has_current_query=False,
    )

    assert plan.status == "Retagging complete in 1.00s."
    assert plan.toast == "Indexed: 5 files / Tagged: 3 / Retagged: 2 (tagger: wd14)"
    assert plan.retag_active is False
    assert plan.run_search is True


def test_plan_index_failed_resets_worker_modes_and_preserves_passthrough() -> None:
    plan = plan_index_failed(
        ONNXRUNTIME_MISSING_MESSAGE,
        refresh_active=False,
        retag_active=False,
        db_display="library.db",
        passthrough_message=ONNXRUNTIME_MISSING_MESSAGE,
    )

    assert plan.status == ONNXRUNTIME_MISSING_MESSAGE
    assert plan.toast == ONNXRUNTIME_MISSING_MESSAGE
    assert plan.refresh_active is False
    assert plan.retag_active is False


def test_connection_retry_action_classifies_restore_errors() -> None:
    assert connection_retry_action(sqlite3.OperationalError("database is locked"), 2) == "retry"
    assert connection_retry_action(sqlite3.OperationalError("database is busy"), 1) == "give_up"
    assert connection_retry_action(sqlite3.OperationalError("disk I/O error"), 2) == "raise"
