"""Tests for the application data path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import AppPaths
from utils import paths


@pytest.mark.parametrize("extra", ["", "subdir"])
def test_get_data_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: str) -> None:
    override = tmp_path / "data" / extra
    monkeypatch.setattr(paths, "_APP_PATHS", AppPaths(env={"KOE_DATA_DIR": str(override)}))
    assert paths.get_data_dir() == override.expanduser()


def test_ensure_dirs_creates_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "custom_data"
    monkeypatch.setattr(paths, "_APP_PATHS", AppPaths(env={"KOE_DATA_DIR": str(data_dir)}))

    target_db = paths.get_db_path()
    cache_dir = paths.get_cache_dir()
    index_dir = paths.get_index_dir()

    assert not target_db.exists()
    assert not cache_dir.exists()
    assert not index_dir.exists()

    paths.ensure_dirs()

    assert target_db.parent.is_dir()
    assert cache_dir.is_dir()
    assert index_dir.is_dir()
