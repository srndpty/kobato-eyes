"""Smoke tests ensuring new PyQt6 enum locations are available."""

from __future__ import annotations

from PyQt6.QtCore import Qt


def test_qt_alignment_flag_access() -> None:
    """Ensure AlignCenter can be referenced from Qt.AlignmentFlag."""

    assert isinstance(Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag)


def test_qt_item_data_roles_accessible() -> None:
    """Ensure required Qt.ItemDataRole members exist."""

    assert isinstance(Qt.ItemDataRole.TextAlignmentRole, Qt.ItemDataRole)
    assert isinstance(Qt.ItemDataRole.UserRole, Qt.ItemDataRole)
    assert isinstance(Qt.ItemDataRole.DecorationRole, Qt.ItemDataRole)
