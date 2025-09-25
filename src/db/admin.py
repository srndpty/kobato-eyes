"""Administrative helpers for managing the kobato-eyes database file."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from db import connection as db_connection
from utils.paths import get_index_dir

logger = logging.getLogger(__name__)


def _resolve_path(db_path: str | Path) -> Path:
    path = Path(db_path).expanduser()
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _copy_backup(source: Path, suffix: str) -> Path:
    destination = source.with_name(f"{source.name}.{suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _format_backup_suffix() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"bak-{timestamp}"


def _reset_bootstrap_cache(original: str | Path, resolved: Path) -> None:
    db_connection._BOOTSTRAPPED.discard(str(original))
    db_connection._BOOTSTRAPPED.discard(str(resolved))


def reset_database(
    db_path: str | Path,
    *,
    backup: bool = True,
    purge_hnsw: bool = True,
) -> dict[str, object]:
    """Reset the SQLite database and associated artifacts.

    The caller must ensure that all active connections to the target database are
    closed before invoking this function.
    """

    resolved_path = _resolve_path(db_path)
    wal_path = resolved_path.with_name(f"{resolved_path.name}-wal")
    shm_path = resolved_path.with_name(f"{resolved_path.name}-shm")

    backup_paths: list[Path] = []
    if backup:
        suffix = _format_backup_suffix()
        for candidate in (resolved_path, wal_path, shm_path):
            if candidate.exists():
                try:
                    backup_paths.append(_copy_backup(candidate, suffix))
                except OSError:
                    logger.exception("Failed to backup %s", candidate)
                    raise

    for candidate in (resolved_path, wal_path, shm_path):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove %s", candidate)
            raise

    hnsw_deleted = False
    if purge_hnsw:
        try:
            index_dir = get_index_dir()
            hnsw_path = index_dir / "hnsw_cosine.bin"
            if hnsw_path.exists():
                hnsw_path.unlink()
                hnsw_deleted = True
        except OSError:
            logger.exception("Failed to remove HNSW index file")
            raise

    _reset_bootstrap_cache(db_path, resolved_path)
    db_connection.bootstrap_if_needed(resolved_path)

    return {
        "db": resolved_path,
        "backup_paths": backup_paths,
        "hnsw_deleted": hnsw_deleted,
    }


__all__ = ["reset_database"]
