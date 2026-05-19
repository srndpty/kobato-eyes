"""Tests for lightweight result delegate helpers."""

from __future__ import annotations

from ui.result_delegates import HighlightDelegate, grid_caption_lines, should_paint_text_background


def test_grid_caption_lines_compacts_extra_lines() -> None:
    assert grid_caption_lines("file.png\ntag_a, tag_b\ntag_c") == ["file.png", "tag_a, tag_b tag_c"]
    assert grid_caption_lines("") == [""]


def test_should_paint_text_background_detects_dark_base() -> None:
    assert should_paint_text_background(0, 0, 0) is True
    assert should_paint_text_background(255, 255, 255) is False


def test_highlight_delegate_ignores_malformed_tag_entries() -> None:
    html = HighlightDelegate._to_html_with_highlight(
        "plain <text>",
        ["nurse"],
        [
            (),
            ("nurse", "bad-score", 0),
            object(),
        ],
        bg="#ffee77",
        fg="#000000",
    )

    assert html == "plain &lt;text&gt;"


def test_highlight_delegate_uses_fallback_text_without_tags() -> None:
    html = HighlightDelegate._to_html_with_highlight(
        "fallback & text",
        ["fallback"],
        None,
        bg="#ffee77",
        fg="#000000",
    )

    assert html == "fallback &amp; text"
