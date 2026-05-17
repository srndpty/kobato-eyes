"""Tests for index feedback formatting helpers."""

from __future__ import annotations

from pathlib import Path

from tagger.wd14_onnx import ONNXRUNTIME_MISSING_MESSAGE
from ui.index_feedback import format_index_failure, format_index_success_toast, format_refresh_feedback


def test_format_refresh_feedback_includes_processed_missing_counts(tmp_path: Path) -> None:
    feedback = format_refresh_feedback(
        {
            "queued": 4,
            "tagged": 2,
            "missing": 5,
            "soft_deleted": 3,
            "hard_deleted": 0,
            "elapsed_sec": 1.25,
        },
        [tmp_path / "library"],
    )

    assert feedback.status == (
        f"Refresh complete: 2 tagged, 3 missing removed (soft delete, 1.25s; queued 4). [{tmp_path / 'library'}]"
    )
    assert feedback.toast == f"2 tagged; 3 missing removed {tmp_path / 'library'}"


def test_format_index_success_toast_includes_retag_counts() -> None:
    message = format_index_success_toast(
        {
            "scanned": 10,
            "tagged": 7,
            "retagged": 3,
            "retagged_marked": 5,
            "tagger_name": "dummy",
        },
        retag_active=True,
    )

    assert message == "Indexed: 10 files / Tagged: 7 / Retagged: 3/5 (tagger: dummy)"


def test_format_index_failure_allows_missing_runtime_passthrough() -> None:
    assert (
        format_index_failure(
            ONNXRUNTIME_MISSING_MESSAGE,
            prefix="Indexing",
            db_display="library.db",
            passthrough_message=ONNXRUNTIME_MISSING_MESSAGE,
        )
        == ONNXRUNTIME_MISSING_MESSAGE
    )


def test_format_index_failure_includes_context_for_regular_errors() -> None:
    assert (
        format_index_failure("boom", prefix="Refreshing", db_display="library.db")
        == "Refreshing failed (DB: library.db): boom"
    )
