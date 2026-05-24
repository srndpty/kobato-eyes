"""Tests for pure tag-result action helpers."""

from __future__ import annotations

from pathlib import Path

from ui.tags_result_actions import (
    build_copy_tags_payload,
    collect_delete_entries,
    normalize_selected_rows,
    result_row_from_stored,
)


def test_result_row_from_stored_accepts_valid_model_or_stored_rows() -> None:
    assert result_row_from_stored(1, None, result_count=3) == 1
    assert result_row_from_stored(1, "2", result_count=3) == 2
    assert result_row_from_stored(1, "bad", result_count=3) is None
    assert result_row_from_stored(9, None, result_count=3) is None


def test_normalize_selected_rows_filters_invalid_and_deduplicates() -> None:
    assert normalize_selected_rows([2, "1", 2, -1, "bad", 4], result_count=3) == [1, 2]


def test_collect_delete_entries_uses_only_rows_with_ids_and_paths(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    rows = [
        {"id": "1", "path": str(image)},
        {"id": None, "path": str(image)},
        {"id": 3, "path": ""},
    ]

    assert collect_delete_entries(rows, [0, 1, 2, 9]) == [(1, image)]


def test_build_copy_tags_payload_formats_plain_and_scored_tags() -> None:
    raw_tags = [("alpha", 0.9, 0), ("low", 0.1, 0), ("character", 0.7, 4)]

    plain = build_copy_tags_payload(raw_tags, include_scores=False)
    scored = build_copy_tags_payload(raw_tags, include_scores=True)

    assert plain is not None
    assert plain.text == "alpha, low, character"
    assert "コピーしました" in plain.feedback
    assert scored is not None
    assert scored.text == "alpha (0.90), low (0.10), character (0.70)"
    assert build_copy_tags_payload([("low", 0.09, 0)], include_scores=False) is None
