"""Tests for duplicate widget formatting and layout helpers."""

from __future__ import annotations

from pathlib import Path

from dup.scanner import DuplicateClusterEntry, DuplicateFile
from ui.dup_widgets import (
    duplicate_tile_metadata,
    format_duplicate_resolution,
    format_duplicate_size,
    thumb_panel_columns,
    thumb_panel_content_height,
)


def _entry(*, size: int | None, width: int | None, height: int | None, hamming: int | None) -> DuplicateClusterEntry:
    file = DuplicateFile(file_id=1, path=Path("a.png"), size=size, width=width, height=height, phash=1)
    return DuplicateClusterEntry(file=file, best_hamming=hamming)


def test_duplicate_metadata_formats_missing_and_present_values() -> None:
    assert format_duplicate_size(None) == "-"
    assert format_duplicate_size(1536) == "1.5 KB"
    assert format_duplicate_resolution(None, 10) == "-"
    assert format_duplicate_resolution(640, 480) == "640×480"
    assert duplicate_tile_metadata(_entry(size=1536, width=640, height=480, hamming=2)) == "1.5 KB   640×480   H:2"
    assert duplicate_tile_metadata(_entry(size=None, width=None, height=None, hamming=None)) == "-   -   -"


def test_thumb_panel_layout_helpers_clamp_to_valid_grid() -> None:
    assert thumb_panel_columns(0, 100, 10) == 1
    assert thumb_panel_columns(350, 100, 10) == 3
    assert thumb_panel_content_height(0, 3, 120, 12) == 0
    assert thumb_panel_content_height(5, 3, 120, 12) == 264
