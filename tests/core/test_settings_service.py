"""Tests for the settings service abstraction."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.config import AppPaths, PipelineSettings, SettingsService


class DummyPlatformDirs:
    def __init__(self, root: Path, name: str) -> None:
        base = root / name
        self.user_data_dir = str(base / "data")
        self.user_config_dir = str(base / "config")


def _make_app_paths(root: Path) -> AppPaths:
    def factory(name: str) -> DummyPlatformDirs:
        return DummyPlatformDirs(root, name)

    return AppPaths(env={}, platform_dirs_factory=factory)


def test_load_returns_defaults_when_missing(tmp_path: Path) -> None:
    app_paths = _make_app_paths(tmp_path)
    service = SettingsService(app_paths)

    loaded = service.load()

    assert isinstance(loaded, PipelineSettings)
    assert loaded == PipelineSettings()


def test_save_and_reload_roundtrip(tmp_path: Path) -> None:
    app_paths = _make_app_paths(tmp_path)
    service = SettingsService(app_paths)

    settings = PipelineSettings(roots=[tmp_path / "images"], index_dir=None)
    service.save(settings)

    config_file = service.config_path
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    assert data["roots"] == [str((tmp_path / "images").expanduser())]
    assert data["index_dir"] == str(app_paths.index_dir())

    reloaded = service.load()
    assert reloaded.roots == settings.roots
    assert reloaded.index_dir is None


def test_invalid_yaml_falls_back_to_defaults(tmp_path: Path) -> None:
    app_paths = _make_app_paths(tmp_path)
    service = SettingsService(app_paths)

    config_file = service.config_path
    config_file.write_text("::not yaml::", encoding="utf-8")

    loaded = service.load()
    assert loaded == PipelineSettings()


def test_non_mapping_payload_falls_back(tmp_path: Path) -> None:
    app_paths = _make_app_paths(tmp_path)
    service = SettingsService(app_paths)

    config_file = service.config_path
    config_file.write_text(yaml.safe_dump(["invalid", "structure"]), encoding="utf-8")

    loaded = service.load()
    assert loaded == PipelineSettings()
