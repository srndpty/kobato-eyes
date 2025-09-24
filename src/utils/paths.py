"""Helpers for resolving application data directories."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from platformdirs import PlatformDirs

logger = logging.getLogger(__name__)

_APP_NAME = "kobato-eyes"
_LEGACY_APP_NAME = "KobatoEyes"
_ENV_VAR = "KOE_DATA_DIR"


def _platform_data_dir(appname: str) -> Path:
    """Return the data directory resolved by :mod:`platformdirs`."""

    dirs = PlatformDirs(appname=appname, appauthor=False, roaming=True)
    return Path(dirs.user_data_dir)


def get_data_dir() -> Path:
    """Return the directory used to persist application data."""

    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser()

    return _platform_data_dir(_APP_NAME)


def get_db_path() -> Path:
    """Return the path to the SQLite database file."""

    return get_data_dir() / "kobato-eyes.db"


def get_index_dir() -> Path:
    """Return the directory used to store search indices."""

    return get_data_dir() / "index"


def get_cache_dir() -> Path:
    """Return the directory used to store cache artifacts."""

    return get_data_dir() / "cache"


def get_log_dir() -> Path:
    """Return the directory used to store application log files."""

    return get_data_dir() / "logs"


def ensure_dirs() -> None:
    """Ensure that the application data directories exist."""

    data_dir = get_data_dir()
    index_dir = get_index_dir()
    cache_dir = get_cache_dir()
    log_dir = get_log_dir()

    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


def migrate_data_dir_if_needed() -> bool:
    """Move data from the legacy directory name to the current one.

    Returns ``True`` when at least one entry was migrated.
    """

    override = os.environ.get(_ENV_VAR)
    if override:
        logger.info(
            "Environment variable %s is set to %s; skipping data directory migration.",
            _ENV_VAR,
            override,
        )
        return False

    target_dir = _platform_data_dir(_APP_NAME)
    legacy_dir = _platform_data_dir(_LEGACY_APP_NAME)

    if os.path.normcase(str(legacy_dir)) == os.path.normcase(str(target_dir)):
        logger.info("Legacy data directory already matches target path %s.", target_dir)
        return False

    legacy_path = Path(legacy_dir)
    target_path = Path(target_dir)

    if not legacy_path.exists():
        logger.info("No legacy data directory found at %s; skipping migration.", legacy_path)
        return False

    if not legacy_path.is_dir():
        logger.info(
            "Legacy data path %s is not a directory; skipping migration.", legacy_path
        )
        return False

    target_path.mkdir(parents=True, exist_ok=True)

    moved_any = False
    skipped: list[str] = []

    for entry in legacy_path.iterdir():
        destination = target_path / entry.name
        if destination.exists():
            skipped.append(entry.name)
            continue
        shutil.move(str(entry), str(destination))
        moved_any = True

    if moved_any:
        logger.info("Migrated data directory from %s to %s.", legacy_path, target_path)
    else:
        logger.info("Legacy directory %s had nothing to migrate.", legacy_path)

    if skipped:
        logger.info(
            "Skipped migrating entries already present in %s: %s",
            target_path,
            ", ".join(sorted(skipped)),
        )

    try:
        legacy_path.rmdir()
    except OSError:
        # Directory still contains skipped items; leave it for manual review.
        pass

    return moved_any


__all__ = [
    "ensure_dirs",
    "get_cache_dir",
    "get_data_dir",
    "get_db_path",
    "get_log_dir",
    "get_index_dir",
    "migrate_data_dir_if_needed",
]
