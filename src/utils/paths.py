"""Helpers for resolving application data directories."""

from __future__ import annotations

from pathlib import Path

from core.config import AppPaths

_APP_PATHS = AppPaths()


def get_app_paths() -> AppPaths:
    """Return the current :class:`AppPaths` instance."""

    return _APP_PATHS


def set_app_paths(app_paths: AppPaths) -> None:
    """Override the global :class:`AppPaths` instance."""

    global _APP_PATHS
    _APP_PATHS = app_paths


def get_data_dir() -> Path:
    """Return the directory used to persist application data."""

    return _APP_PATHS.data_dir()


def get_db_path() -> Path:
    """Return the path to the SQLite database file."""

    return _APP_PATHS.db_path()


def get_index_dir() -> Path:
    """Return the directory used to store search indices."""

    return _APP_PATHS.index_dir()


def get_cache_dir() -> Path:
    """Return the directory used to store cache artefacts."""

    return _APP_PATHS.cache_dir()


def get_log_dir() -> Path:
    """Return the directory used to store application log files."""

    return _APP_PATHS.log_dir()


def ensure_dirs() -> None:
    """Ensure that the application data directories exist."""

    _APP_PATHS.ensure_data_dirs()


def migrate_data_dir_if_needed() -> bool:
    """Move data from the legacy directory name to the current one."""

    return _APP_PATHS.migrate_data_dir_if_needed()


__all__ = [
    "ensure_dirs",
    "get_app_paths",
    "get_cache_dir",
    "get_data_dir",
    "get_db_path",
    "get_index_dir",
    "get_log_dir",
    "migrate_data_dir_if_needed",
    "set_app_paths",
]
