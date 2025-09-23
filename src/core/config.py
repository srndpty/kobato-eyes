"""Persistent configuration handling for kobato-eyes."""

from __future__ import annotations

import os
from pathlib import Path

from core.settings import PipelineSettings

APP_DIR_NAME = "kobato-eyes"
CONFIG_FILENAME = "config.yaml"


def _app_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_dir = base / APP_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def config_path() -> Path:
    return _app_data_dir() / CONFIG_FILENAME
    

def load_settings() -> PipelineSettings:
    path = config_path()
    if not path.exists():
        return PipelineSettings()

    return PipelineSettings.load(path)


def save_settings(settings: PipelineSettings) -> None:
    path = config_path()
    settings.save(path)


__all__ = ["config_path", "load_settings", "save_settings"]
