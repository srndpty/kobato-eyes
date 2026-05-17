"""Tests for spinner overlay pure helpers."""

from __future__ import annotations

from PyQt6.QtCore import QRect

from ui.widgets.spinner_overlay import DEFAULT_OVERLAY_MESSAGE, overlay_geometry, overlay_message


def test_overlay_message_uses_default_only_for_none() -> None:
    assert overlay_message(None) == DEFAULT_OVERLAY_MESSAGE
    assert overlay_message("") == ""
    assert overlay_message("Indexing") == "Indexing"


def test_overlay_geometry_handles_missing_parent() -> None:
    assert overlay_geometry(None) == QRect()
