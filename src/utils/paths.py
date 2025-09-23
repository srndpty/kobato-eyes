"""Helpers for resolving application data directories."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

_APP_NAME = "KobatoEyes"
_ENV_VAR = "KOE_DATA_DIR"


def get_data_dir() -> Path:
    """Return the directory used to persist application data."""

    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser()

    dirs = PlatformDirs(appname=_APP_NAME, appauthor=False, roaming=True)
    return Path(dirs.user_data_dir)


def get_db_path() -> Path:
    """Return the path to the SQLite database file."""

    return get_data_dir() / "kobato-eyes.db"


def get_index_dir() -> Path:
    """Return the directory used to store search indices."""

    return get_data_dir() / "index"


def get_cache_dir() -> Path:
    """Return the directory used to store cache artifacts."""

    return get_data_dir() / "cache"


def ensure_dirs() -> None:
    """Ensure that the application data directories exist."""

    data_dir = get_data_dir()
    index_dir = get_index_dir()
    cache_dir = get_cache_dir()

    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ensure_dirs",
    "get_cache_dir",
    "get_data_dir",
    "get_db_path",
    "get_index_dir",
]
