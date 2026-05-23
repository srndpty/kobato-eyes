"""Tests for pure tag-result UI state helpers."""

from __future__ import annotations

from pathlib import Path

from ui.tags_delete_state import (
    format_delete_confirmation,
    format_delete_failure_reason,
    format_delete_result_status,
    format_deleting_status,
)
from ui.tags_result_state import (
    coerce_file_id,
    coerce_result_path,
    plan_result_removal,
    should_queue_missing_thumbnail,
    thumbnail_matches_result,
)


def test_thumbnail_match_and_coercion_helpers(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    rows = [{"id": "10", "path": str(image)}, {"id": 11, "path": "  "}]

    assert coerce_file_id("10") == 10
    assert coerce_file_id(None) is None
    assert coerce_result_path(str(image)) == image
    assert coerce_result_path("") is None
    assert thumbnail_matches_result(rows, row=0, file_id=10)
    assert not thumbnail_matches_result(rows, row=1, file_id=10)
    assert not thumbnail_matches_result(rows, row=2, file_id=10)


def test_plan_result_removal_tracks_offset_and_selection() -> None:
    rows = [{"id": 1}, {"id": "2"}, {"id": 3}, {"id": 4}]

    plan = plan_result_removal(rows, [2, 3], offset_file_ids=[3])

    assert plan is not None
    assert plan.rows == [1, 2]
    assert plan.offset_removed == 1
    assert plan.next_selection == 1
    assert plan_result_removal(rows, [], offset_file_ids=[]) is None
    assert plan_result_removal(rows, [99], offset_file_ids=[]) is None


def test_should_queue_missing_thumbnail_requires_missing_decoration_and_existing_path(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    assert should_queue_missing_thumbnail({"id": 5, "path": str(image)}, has_thumbnail=False) == (5, image)
    assert should_queue_missing_thumbnail({"id": 5, "path": str(image)}, has_thumbnail=True) is None
    assert should_queue_missing_thumbnail({"id": 5, "path": str(tmp_path / "missing.png")}, has_thumbnail=False) is None


def test_delete_status_helpers_are_stable() -> None:
    path = Path("image.png")

    assert "Move this image" in format_delete_confirmation([path])
    assert format_deleting_status([path]) == "Deleting image.png..."
    assert format_delete_result_status([], [], 1, 3, "tag") == "Delete failed. Showing 3 result(s) for 'tag'"
    assert format_delete_result_status([(1, str(path))], [], 1, 2, "tag") == (
        "Deleted image.png. Showing 2 result(s) for 'tag'"
    )
    assert format_delete_failure_reason("db", "locked") == "moved to trash, but DB update failed: locked"
