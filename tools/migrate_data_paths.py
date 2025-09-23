"""Migrate persisted data into the unified user data directory."""

from __future__ import annotations

import shutil
from pathlib import Path

from utils.paths import ensure_dirs, get_data_dir, get_db_path

_LEGACY_BASENAME = "kobato-eyes.db"
_LEGACY_SUFFIXES = ("", "-wal", "-shm")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def migrate_legacy_data() -> bool:
    """Move legacy database files into the current data directory."""

    ensure_dirs()
    target_db = get_db_path()
    legacy_root = _repo_root()
    legacy_db = legacy_root / _LEGACY_BASENAME

    if not legacy_db.exists():
        return False

    if target_db.exists():
        return False

    target_db.parent.mkdir(parents=True, exist_ok=True)

    for suffix in _LEGACY_SUFFIXES:
        source = legacy_root / f"{_LEGACY_BASENAME}{suffix}"
        if source.exists():
            destination = target_db.with_name(f"{_LEGACY_BASENAME}{suffix}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

    return True


def main() -> None:
    moved = migrate_legacy_data()
    if moved:
        print(f"Database migrated to {get_db_path()}")
    else:
        print(f"No migration required. Data directory is {get_data_dir()}")


if __name__ == "__main__":
    main()
