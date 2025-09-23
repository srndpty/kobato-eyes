"""Shared settings structures for kobato-eyes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

DEFAULT_EXCLUDED = [
    str(Path.home() / "AppData"),
    "C:/Windows",
    "C:/Program Files",
    "C:/Program Files (x86)",
]
DEFAULT_ALLOW_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".gif",
}
DEFAULT_TAG_THRESHOLDS = {
    "general": 0.35,
    "character": 0.25,
    "copyright": 0.25,
}


def _default_excluded() -> list[str]:
    return [str(Path(path)) for path in DEFAULT_EXCLUDED]


def _default_allow_exts() -> set[str]:
    return set(DEFAULT_ALLOW_EXTS)


def default_index_dir() -> str:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return str((base / "KobatoEyes" / "index").resolve())


@dataclass
class TaggerSettings:
    name: str = "dummy"
    model_path: str | None = None
    thresholds: dict[str, float] = field(default_factory=lambda: DEFAULT_TAG_THRESHOLDS.copy())


@dataclass
class EmbedModel:
    name: str = "ViT-L-14"
    device: str = "cuda"
    dim: int = 768

    @property
    def pretrained(self) -> str:
        defaults = {
            "ViT-L-14": "laion2b_s32b_b79k",
            "ViT-H-14": "laion2b_s32b_b79k",
            "RN50": "openai",
        }
        return defaults.get(self.name, "openai")


@dataclass
class PipelineSettings:
    roots: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=_default_excluded)
    allow_exts: set[str] = field(default_factory=_default_allow_exts)
    batch_size: int = 8
    hamming_threshold: int = 10
    cosine_threshold: float = 0.90
    ssim_threshold: float = 0.92
    tagger: TaggerSettings = field(default_factory=TaggerSettings)
    embed_model: EmbedModel = field(default_factory=EmbedModel)
    index_dir: str | None = None

    def __post_init__(self) -> None:
        self.roots = [self._normalise_path(path) for path in self.roots if path]
        self.excluded = [self._normalise_path(path) for path in (self.excluded or [])]
        if not self.excluded:
            self.excluded = _default_excluded()
        self.allow_exts = {self._normalise_ext(ext) for ext in (self.allow_exts or _default_allow_exts()) if ext}
        if self.batch_size <= 0:
            self.batch_size = 1
        if self.index_dir is not None and self.index_dir != "":
            self.index_dir = self._normalise_path(self.index_dir)
        else:
            self.index_dir = None

    @staticmethod
    def _normalise_path(path: str | os.PathLike[str]) -> str:
        return str(Path(path).expanduser())

    @staticmethod
    def _normalise_ext(extension: str) -> str:
        ext = extension.strip().lower()
        if not ext:
            return ext
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext

    @property
    def model_name(self) -> str:
        return self.embed_model.name

    def resolved_index_dir(self) -> str:
        return self.index_dir or default_index_dir()


def normalise_roots(values: Sequence[str | os.PathLike[str]]) -> list[str]:
    return [str(Path(value).expanduser()) for value in values]


def normalise_excluded(values: Sequence[str | os.PathLike[str]]) -> list[str]:
    if not values:
        return _default_excluded()
    return [str(Path(value).expanduser()) for value in values]


__all__ = [
    "PipelineSettings",
    "TaggerSettings",
    "EmbedModel",
    "default_index_dir",
    "normalise_roots",
    "normalise_excluded",
]
