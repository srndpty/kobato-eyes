"""Persistent configuration handling for kobato-eyes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

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
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    roots = [Path(p) for p in data.get("roots", [])]
    excluded = [Path(p) for p in data.get("excluded", [])]
    return PipelineSettings(
        roots=roots,
        excluded=excluded,
        hamming_threshold=int(data.get("hamming_threshold", 8)),
        cosine_threshold=float(data.get("cosine_threshold", 0.2)),
        ssim_threshold=float(data.get("ssim_threshold", 0.9)),
        model_name=str(data.get("model_name", "clip-vit")),
    )


def save_settings(settings: PipelineSettings) -> None:
    payload: dict[str, Any] = {
        "roots": [str(path) for path in settings.roots],
        "excluded": [str(path) for path in settings.excluded],
        "hamming_threshold": settings.hamming_threshold,
        "cosine_threshold": settings.cosine_threshold,
        "ssim_threshold": settings.ssim_threshold,
        "model_name": settings.model_name,
    }
    path = config_path()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


__all__ = ["config_path", "load_settings", "save_settings"]
