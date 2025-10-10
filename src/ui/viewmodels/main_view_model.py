"""ViewModel coordinating application bootstrap and settings handling."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from core.config import load_settings, save_settings
from core.config.schema import PipelineSettings
from db.connection import bootstrap_if_needed, get_conn
from utils.paths import ensure_dirs, get_db_path, migrate_data_dir_if_needed

from .dup_view_model import DupViewModel
from .settings_view_model import SettingsViewModel
from .tags_view_model import TagsViewModel

logger = logging.getLogger(__name__)


def _quick_settle_sqlite(db_path: Path) -> None:
    """Issue pragmatic pragmas to keep SQLite responsive."""

    try:
        conn = get_conn(db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=2000")
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA optimize")
        finally:
            conn.close()
    except Exception:  # pragma: no cover - best effort logging
        logger.exception("Failed to settle SQLite database at %s", db_path)


class MainViewModel(QObject):
    """ViewModel coordinating startup tasks for :class:`ui.app.MainWindow`."""

    settings_changed = pyqtSignal(PipelineSettings)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        migrate_data_dir_if_needed()
        ensure_dirs()
        self._db_path = get_db_path()
        logger.info("DB at %s", self._db_path)
        _quick_settle_sqlite(self._db_path)
        bootstrap_if_needed(self._db_path)
        self._current_settings = load_settings()

    @property
    def db_path(self) -> Path:
        """Return the path to the active SQLite database."""

        return Path(self._db_path)

    @property
    def current_settings(self) -> PipelineSettings:
        """Return the most recently persisted pipeline settings."""

        return self._current_settings

    def create_tags_view_model(self, parent: QObject | None = None) -> TagsViewModel:
        """Return a tags tab view model bound to the managed database."""

        return TagsViewModel(parent, db_path=self.db_path)

    def create_dup_view_model(self, parent: QObject | None = None) -> DupViewModel:
        """Return a duplicates tab view model bound to the managed database."""

        return DupViewModel(parent, db_path=self.db_path)

    def create_settings_view_model(self, parent: QObject | None = None) -> SettingsViewModel:
        """Return a settings view model configured with the persisted settings."""

        view_model = SettingsViewModel(parent, db_path=self.db_path)
        view_model.set_current_settings(self._current_settings)
        view_model.settings_applied.connect(self.apply_settings)
        return view_model

    def apply_settings(self, settings: PipelineSettings) -> None:
        """Persist new settings and notify listeners."""

        save_settings(settings)
        self._current_settings = settings
        self.settings_changed.emit(settings)
