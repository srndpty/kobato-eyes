"""Smoke tests ensuring new PyQt6 enum locations are available."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)

from PyQt6.QtCore import Qt

pytestmark = pytest.mark.gui


def test_qt_alignment_flag_access() -> None:
    """Ensure AlignCenter can be referenced from Qt.AlignmentFlag."""

    assert isinstance(Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag)


def test_qt_item_data_roles_accessible() -> None:
    """Ensure required Qt.ItemDataRole members exist."""

    assert isinstance(Qt.ItemDataRole.TextAlignmentRole, Qt.ItemDataRole)
    assert isinstance(Qt.ItemDataRole.UserRole, Qt.ItemDataRole)
    assert isinstance(Qt.ItemDataRole.DecorationRole, Qt.ItemDataRole)
