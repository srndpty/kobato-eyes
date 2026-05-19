"""Tests for spinner overlay pure helpers."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QRect

from ui.widgets.spinner_overlay import DEFAULT_OVERLAY_MESSAGE, SpinnerOverlay, overlay_geometry, overlay_message


def test_overlay_message_uses_default_only_for_none() -> None:
    assert overlay_message(None) == DEFAULT_OVERLAY_MESSAGE
    assert overlay_message("") == ""
    assert overlay_message("Indexing") == "Indexing"


def test_overlay_geometry_handles_missing_parent() -> None:
    assert overlay_geometry(None) == QRect()


def test_spinner_overlay_updates_message_and_tracks_parent(qtbot) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(320, 200)
    parent.show()
    overlay = SpinnerOverlay(parent)
    qtbot.addWidget(overlay)

    overlay.show(None)
    assert overlay.isVisible()
    assert overlay._message_label.text() == DEFAULT_OVERLAY_MESSAGE
    assert overlay.geometry() == parent.rect()

    overlay.set_message("Indexing")
    assert overlay._message_label.text() == "Indexing"

    parent.resize(480, 240)
    overlay.eventFilter(parent, QEvent(QEvent.Type.Resize))
    assert overlay.geometry() == parent.rect()
