"""Tests for migrating legacy data directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import paths


def test_migrate_data_dir_moves_legacy_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure legacy data directories are migrated without overwriting files."""

    monkeypatch.delenv("KOE_DATA_DIR", raising=False)

    legacy_dir = tmp_path / "KobatoEyes"
    target_dir = tmp_path / "kobato-eyes"
    legacy_dir.mkdir()
    target_dir.mkdir()

    class DummyPlatformDirs:
        def __init__(
            self, appname: str, appauthor: bool = False, roaming: bool = True
        ) -> None:
            mapping = {
                "kobato-eyes": target_dir,
                "KobatoEyes": legacy_dir,
            }
            self.user_data_dir = str(mapping[appname])

    monkeypatch.setattr(paths, "PlatformDirs", DummyPlatformDirs)

    (legacy_dir / "kobato-eyes.db").write_text("legacy-db")
    (legacy_dir / "kobato-eyes.db-wal").write_text("legacy-wal")
    (legacy_dir / "kobato-eyes.db-shm").write_text("legacy-shm")
    (legacy_dir / "config.yaml").write_text("legacy-config")

    index_dir = legacy_dir / "index"
    index_dir.mkdir()
    (index_dir / "entries").write_text("index-data")

    cache_dir = legacy_dir / "cache"
    cache_dir.mkdir()
    (cache_dir / "item.bin").write_text("cache-data")

    existing_config = target_dir / "config.yaml"
    existing_config.write_text("current-config")

    moved = paths.migrate_data_dir_if_needed()

    assert moved is True
    assert (target_dir / "kobato-eyes.db").read_text() == "legacy-db"
    assert (target_dir / "kobato-eyes.db-wal").read_text() == "legacy-wal"
    assert (target_dir / "kobato-eyes.db-shm").read_text() == "legacy-shm"
    assert (target_dir / "index" / "entries").read_text() == "index-data"
    assert (target_dir / "cache" / "item.bin").read_text() == "cache-data"
    assert existing_config.read_text() == "current-config"
    assert (legacy_dir / "config.yaml").read_text() == "legacy-config"
    assert not (legacy_dir / "index").exists()
    assert not (legacy_dir / "cache").exists()

    moved_again = paths.migrate_data_dir_if_needed()

    assert moved_again is False
    assert (target_dir / "kobato-eyes.db").read_text() == "legacy-db"
    assert (legacy_dir / "config.yaml").read_text() == "legacy-config"
