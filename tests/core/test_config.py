"""Tests for persistent configuration handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import AppPaths, PipelineSettings, config_path, configure, load_settings, save_settings
from utils.paths import set_app_paths


class _DummyDirs:
    def __init__(self, root: Path) -> None:
        self.user_data_dir = str(root / "data")
        self.user_config_dir = str(root / "config")


def _make_app_paths(root: Path) -> AppPaths:
    def factory(_: str) -> _DummyDirs:
        return _DummyDirs(root)

    return AppPaths(env={}, platform_dirs_factory=factory)


def test_config_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app_paths = _make_app_paths(tmp_path)
    configure(app_paths)
    set_app_paths(app_paths)
    path = config_path()
    if path.exists():
        path.unlink()

    settings = load_settings()
    assert settings.roots == []

    updated = PipelineSettings(
        roots=[tmp_path / "root"],
        excluded=[tmp_path / "skip"],
        hamming_threshold=5,
    )
    save_settings(updated)

    reloaded = load_settings()
    assert reloaded.roots == updated.roots
    assert reloaded.excluded == updated.excluded
    assert reloaded.hamming_threshold == 5
