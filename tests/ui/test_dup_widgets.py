"""Tests for duplicate widget formatting and layout helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QRect, QSize
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QWidget

from dup.scanner import DuplicateClusterEntry, DuplicateFile
from ui.dup_widgets import (
    ThumbPanel,
    ThumbTile,
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


@pytest.mark.gui
def test_thumb_tile_state_and_pixmap_handling(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    parent = QWidget()
    parent.setProperty("keeper_id", 1)
    qtbot.addWidget(parent)
    tile = ThumbTile(_entry(size=100, width=20, height=10, hamming=1), QSize(32, 32), parent)
    qtbot.addWidget(tile)

    assert tile.is_checked() is False
    tile.set_checked(True)
    assert tile.is_checked() is True
    assert "H:1" in tile.meta1.text()
    assert tile.path == Path("a.png")

    placeholder_pixmap = QPixmap(32, 32)
    placeholder_pixmap.fill(QColor("red"))
    tile.set_pixmap(None, QIcon(placeholder_pixmap))
    assert tile.thumb.pixmap() is not None

    source = QPixmap(12, 20)
    source.fill(QColor("blue"))
    tile.set_pixmap(source, QIcon(placeholder_pixmap))
    assert tile.thumb.pixmap() is not None


@pytest.mark.gui
def test_thumb_panel_relayout_and_visible_tiles(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    entries = [
        _entry(size=100, width=20, height=10, hamming=1),
        _entry(size=200, width=30, height=20, hamming=2),
        _entry(size=300, width=40, height=30, hamming=3),
    ]
    panel = ThumbPanel(entries, keeper_id=1, icon_size=QSize(32, 32), col_gap=4, row_gap=6)
    qtbot.addWidget(panel)
    panel.resize(200, 120)

    panel._relayout()

    assert panel.sizeHint().height() == panel._content_h
    assert len(panel.tiles) == 3
    visible = panel.visible_tiles_in(panel, QRect(0, 0, 200, 120))
    assert visible == panel.tiles
