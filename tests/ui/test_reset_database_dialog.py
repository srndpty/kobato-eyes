"""Smoke tests for the ResetDatabaseDialog."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)

from ui.settings_tab import ResetDatabaseDialog

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.gui


def test_reset_database_dialog_defaults(qtbot: QtBot, tmp_path: Path) -> None:
    """The dialog should construct and enable backup/index options by default."""

    dialog = ResetDatabaseDialog(tmp_path / "example.db")
    qtbot.addWidget(dialog)

    assert dialog.backup_enabled is True
    assert dialog.start_index_enabled is True
