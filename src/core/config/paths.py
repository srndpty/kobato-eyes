"""Path resolution helpers for kobato-eyes configuration and data."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from platformdirs import PlatformDirs

logger = logging.getLogger(__name__)


class AppPaths:
    """Resolve application directories with support for dependency injection."""

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        *,
        app_name: str = "kobato-eyes",
        legacy_app_name: str = "KobatoEyes",
        env_var: str = "KOE_DATA_DIR",
        platform_dirs_factory: Callable[[str], PlatformDirs] | None = None,
    ) -> None:
        self._env = MappingProxyType(dict(env) if env is not None else dict(os.environ))
        self._app_name = app_name
        self._legacy_app_name = legacy_app_name
        self._env_var = env_var
        self._platform_dirs_factory = platform_dirs_factory or self._default_platform_dirs

    @staticmethod
    def _default_platform_dirs(app_name: str) -> PlatformDirs:
        return PlatformDirs(appname=app_name, appauthor=False, roaming=True)

    def _platform_dirs(self) -> PlatformDirs:
        return self._platform_dirs_factory(self._app_name)

    def _legacy_platform_dirs(self) -> PlatformDirs:
        return self._platform_dirs_factory(self._legacy_app_name)

    def data_dir(self) -> Path:
        """Return the directory used to persist application data."""

        override = self._env.get(self._env_var)
        if override:
            return Path(override).expanduser()
        return Path(self._platform_dirs().user_data_dir)

    def config_dir(self) -> Path:
        """Return the directory used to persist configuration files."""

        return Path(self._platform_dirs().user_config_dir)

    def config_path(self, filename: str = "config.yaml") -> Path:
        """Return the full path to the configuration file."""

        return self.config_dir() / filename

    def db_path(self) -> Path:
        """Return the path to the SQLite database file."""

        return self.data_dir() / "kobato-eyes.db"

    def index_dir(self) -> Path:
        """Return the directory used for search indices."""

        return self.data_dir() / "index"

    def cache_dir(self) -> Path:
        """Return the directory used for cache artefacts."""

        return self.data_dir() / "cache"

    def log_dir(self) -> Path:
        """Return the directory used to store log files."""

        return self.data_dir() / "logs"

    def ensure_data_dirs(self) -> None:
        """Ensure that all data directories exist."""

        for directory in (self.data_dir(), self.index_dir(), self.cache_dir(), self.log_dir()):
            directory.mkdir(parents=True, exist_ok=True)

    def migrate_data_dir_if_needed(self) -> bool:
        """Migrate data from a legacy directory name if required."""

        override = self._env.get(self._env_var)
        if override:
            logger.info(
                "Environment variable %s is set to %s; skipping data directory migration.",
                self._env_var,
                override,
            )
            return False

        target_dir = Path(self._platform_dirs().user_data_dir)
        legacy_dir = Path(self._legacy_platform_dirs().user_data_dir)

        if os.path.normcase(str(target_dir)) == os.path.normcase(str(legacy_dir)):
            logger.info("Legacy data directory already matches target path %s.", target_dir)
            return False

        if not legacy_dir.exists():
            logger.info("No legacy data directory found at %s; skipping migration.", legacy_dir)
            return False

        if not legacy_dir.is_dir():
            logger.info("Legacy data path %s is not a directory; skipping migration.", legacy_dir)
            return False

        target_dir.mkdir(parents=True, exist_ok=True)

        moved_any = False
        skipped: list[str] = []

        for entry in legacy_dir.iterdir():
            destination = target_dir / entry.name
            if destination.exists():
                skipped.append(entry.name)
                continue
            shutil.move(str(entry), str(destination))
            moved_any = True

        if moved_any:
            logger.info("Migrated data directory from %s to %s.", legacy_dir, target_dir)
        else:
            logger.info("Legacy directory %s had nothing to migrate.", legacy_dir)

        if skipped:
            logger.info(
                "Skipped migrating entries already present in %s: %s",
                target_dir,
                ", ".join(sorted(skipped)),
            )

        try:
            legacy_dir.rmdir()
        except OSError:
            pass

        return moved_any


__all__ = ["AppPaths"]
