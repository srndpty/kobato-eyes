"""Tests for pure tag-result delete state helpers."""

from __future__ import annotations

from pathlib import Path

from ui.tags_delete_state import (
    format_delete_confirmation,
    format_delete_failure_reason,
    format_delete_result_status,
    format_deleting_status,
)


def test_delete_status_helpers_are_stable() -> None:
    path = Path("image.png")

    assert "Move this image" in format_delete_confirmation([path])
    assert format_deleting_status([path]) == "Deleting image.png..."
    assert format_delete_result_status([], [], 1, 3, "tag") == "Delete failed. Showing 3 result(s) for 'tag'"
    assert format_delete_result_status([(1, str(path))], [], 1, 2, "tag") == (
        "Deleted image.png. Showing 2 result(s) for 'tag'"
    )
    assert format_delete_failure_reason("db", "locked") == "moved to trash, but DB update failed: locked"
