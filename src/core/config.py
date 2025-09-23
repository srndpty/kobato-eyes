"""Persistent configuration handling for kobato-eyes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from core.settings import (
    EmbedModel,
    PipelineSettings,
    TaggerSettings,
    default_index_dir,
    normalise_excluded,
    normalise_roots,
)

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


def _normalise_exts(values: Any, fallback: set[str]) -> set[str]:
    if not values:
        return set(fallback)
    normalised: set[str] = set()
    for value in values:
        ext = str(value).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalised.add(ext)
    return normalised or set(fallback)


def load_settings() -> PipelineSettings:
    path = config_path()
    defaults = PipelineSettings()
    if not path.exists():
        return defaults

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    roots = normalise_roots(data.get("roots", defaults.roots))

    excluded_source = data.get("excluded")
    if excluded_source is None and "excludes" in data:
        excluded_source = data["excludes"]
    excluded = normalise_excluded(excluded_source or defaults.excluded)

    allow_exts = _normalise_exts(data.get("allow_exts"), defaults.allow_exts)

    batch_size = int(data.get("batch_size", defaults.batch_size))
    hamming_threshold = int(data.get("hamming_threshold", defaults.hamming_threshold))
    cosine_threshold = float(data.get("cosine_threshold", defaults.cosine_threshold))
    ssim_threshold = float(data.get("ssim_threshold", defaults.ssim_threshold))

    tagger_conf = data.get("tagger", {}) or {}
    tagger_thresholds = defaults.tagger.thresholds.copy()
    tagger_thresholds.update({str(key): float(value) for key, value in (tagger_conf.get("thresholds") or {}).items()})
    tagger = TaggerSettings(
        name=str(tagger_conf.get("name", defaults.tagger.name)),
        model_path=tagger_conf.get("model_path", defaults.tagger.model_path),
        thresholds=tagger_thresholds,
    )

    embed_conf = data.get("embed_model", {}) or {}
    embed_name = str(embed_conf.get("name", data.get("model_name", defaults.embed_model.name)))
    embed_device = str(embed_conf.get("device", defaults.embed_model.device))
    embed_dim = int(embed_conf.get("dim", embed_conf.get("dims", defaults.embed_model.dim)))
    embed_model = EmbedModel(name=embed_name, device=embed_device, dim=embed_dim)

    index_dir_value = data.get("index_dir")
    if index_dir_value in (None, ""):
        index_dir = None
    else:
        index_dir = str(Path(index_dir_value).expanduser())

    settings = PipelineSettings(
        roots=roots,
        excluded=excluded,
        allow_exts=allow_exts,
        batch_size=batch_size,
        hamming_threshold=hamming_threshold,
        cosine_threshold=cosine_threshold,
        ssim_threshold=ssim_threshold,
        tagger=tagger,
        embed_model=embed_model,
        index_dir=index_dir,
    )
    return settings


def save_settings(settings: PipelineSettings) -> None:
    payload: dict[str, Any] = {
        "roots": [str(path) for path in settings.roots],
        "excluded": [str(path) for path in settings.excluded],
        "allow_exts": sorted(settings.allow_exts),
        "batch_size": settings.batch_size,
        "hamming_threshold": settings.hamming_threshold,
        "cosine_threshold": settings.cosine_threshold,
        "ssim_threshold": settings.ssim_threshold,
        "tagger": {
            "name": settings.tagger.name,
            "model_path": settings.tagger.model_path,
            "thresholds": settings.tagger.thresholds,
        },
        "embed_model": {
            "name": settings.embed_model.name,
            "device": settings.embed_model.device,
            "dim": settings.embed_model.dim,
        },
        "index_dir": settings.index_dir or default_index_dir(),
    }
    path = config_path()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


__all__ = ["config_path", "load_settings", "save_settings"]
