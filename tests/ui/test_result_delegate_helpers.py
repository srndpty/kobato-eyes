"""Tests for lightweight result delegate helpers."""

from __future__ import annotations

from ui.result_delegates import grid_caption_lines, should_paint_text_background


def test_grid_caption_lines_compacts_extra_lines() -> None:
    assert grid_caption_lines("file.png\ntag_a, tag_b\ntag_c") == ["file.png", "tag_a, tag_b tag_c"]
    assert grid_caption_lines("") == [""]


def test_should_paint_text_background_detects_dark_base() -> None:
    assert should_paint_text_background(0, 0, 0) is True
    assert should_paint_text_background(255, 255, 255) is False
