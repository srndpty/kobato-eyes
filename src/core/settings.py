"""Shared settings structures for kobato-eyes."""

from __future__ import annotations

import os
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from utils.paths import ensure_dirs, get_index_dir

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
    ensure_dirs()
    return str(get_index_dir())


@dataclass
class TaggerSettings:
    name: str = "dummy"
    model_path: str | None = None
    tags_csv: str | None = None
    thresholds: dict[str, float] = field(default_factory=lambda: DEFAULT_TAG_THRESHOLDS.copy())


_DEFAULT_PRETRAINED_TAGS = {
    "ViT-L-14": "openai",
    "ViT-H-14": "laion2b_s32b_b82k",
    "RN50": "openai",
}


_VALID_EMBED_DEVICES = {"auto", "cuda", "cpu"}


def _default_pretrained(model_name: str) -> str:
    return _DEFAULT_PRETRAINED_TAGS.get(model_name, "openai")


@dataclass
class EmbedModel:
    name: str = "ViT-L-14"
    pretrained: str = "openai"
    device: str = "auto"
    dim: int = 768

    def __post_init__(self) -> None:
        if not isinstance(self.pretrained, str):
            self.pretrained = "" if self.pretrained is None else str(self.pretrained)
        self.pretrained = self.pretrained.strip()
        if not self.pretrained:
            self.pretrained = _default_pretrained(self.name)
        device_value = (self.device or "auto") if isinstance(self.device, str) else str(self.device)
        device_normalised = device_value.strip().lower()
        if device_normalised not in _VALID_EMBED_DEVICES:
            device_normalised = "auto"
        self.device = device_normalised


@dataclass
class PipelineSettings:
    roots: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=_default_excluded)
    allow_exts: set[str] = field(default_factory=_default_allow_exts)
    auto_index: bool = True
    batch_size: int = 8
    model_name_init: InitVar[str | None] = None
    hamming_threshold: int = 10
    cosine_threshold: float = 0.90
    ssim_threshold: float = 0.92
    tagger: TaggerSettings = field(default_factory=TaggerSettings)
    embed_model: EmbedModel = field(default_factory=EmbedModel)
    index_dir: str | None = None

    def __post_init__(self, model_name_init: str | None = None) -> None:
        self.roots = [self._normalise_path(path) for path in self.roots if path]
        self.excluded = [self._normalise_path(path) for path in (self.excluded or [])]
        if not self.excluded:
            self.excluded = _default_excluded()
        self.allow_exts = {self._normalise_ext(ext) for ext in (self.allow_exts or _default_allow_exts()) if ext}
        self.auto_index = bool(self.auto_index)
        if self.batch_size <= 0:
            self.batch_size = 1
        if model_name_init:
            self.embed_model.name = str(model_name_init)
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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PipelineSettings:
        defaults = cls()

        raw_roots = data.get("roots")
        roots = normalise_roots(raw_roots or defaults.roots)

        excluded_source = data.get("excluded")
        if excluded_source is None and "excludes" in data:
            excluded_source = data["excludes"]
        excluded = normalise_excluded(excluded_source or defaults.excluded)

        allow_exts = _normalise_exts(data.get("allow_exts"), defaults.allow_exts)

        batch_size = _coerce_int(data.get("batch_size"), defaults.batch_size)
        # auto_index = _coerce_bool(data.get("auto_index"), defaults.auto_index)
        hamming_threshold = _coerce_int(data.get("hamming_threshold"), defaults.hamming_threshold)
        cosine_threshold = _coerce_float(data.get("cosine_threshold"), defaults.cosine_threshold)
        ssim_threshold = _coerce_float(data.get("ssim_threshold"), defaults.ssim_threshold)

        tagger_conf = data.get("tagger", {}) or {}
        tagger_thresholds = defaults.tagger.thresholds.copy()
        for key, value in (tagger_conf.get("thresholds") or {}).items():
            try:
                tagger_thresholds[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        tagger_model = _coerce_optional_path(tagger_conf.get("model_path", defaults.tagger.model_path))
        tagger_csv = _coerce_optional_path(tagger_conf.get("tags_csv", defaults.tagger.tags_csv))
        tagger = TaggerSettings(
            name=str(tagger_conf.get("name", defaults.tagger.name)),
            model_path=tagger_model,
            tags_csv=tagger_csv,
            thresholds=tagger_thresholds,
        )

        embed_conf = data.get("embed_model", {}) or {}
        embed_name = embed_conf.get("name")
        if embed_name is None and "model_name" in data:
            embed_name = data["model_name"]
        embed_pretrained = _coerce_str(embed_conf.get("pretrained"))
        embed_model = EmbedModel(
            name=str(embed_name or defaults.embed_model.name),
            pretrained=embed_pretrained,
            device=str(embed_conf.get("device", defaults.embed_model.device)),
            dim=_coerce_int(embed_conf.get("dim", embed_conf.get("dims")), defaults.embed_model.dim),
        )

        index_dir_value = data.get("index_dir")
        if index_dir_value in (None, ""):
            index_dir = None
        else:
            index_dir = cls._normalise_path(index_dir_value)

        return cls(
            roots=roots,
            excluded=excluded,
            allow_exts=allow_exts,
            auto_index=False,
            batch_size=batch_size,
            hamming_threshold=hamming_threshold,
            cosine_threshold=cosine_threshold,
            ssim_threshold=ssim_threshold,
            tagger=tagger,
            embed_model=embed_model,
            index_dir=index_dir,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> PipelineSettings:
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raw = {}
        return cls.from_mapping(raw)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "roots": [str(path) for path in self.roots],
            "excluded": [str(path) for path in self.excluded],
            "allow_exts": sorted(self.allow_exts),
            "auto_index": bool(self.auto_index),
            "batch_size": self.batch_size,
            "hamming_threshold": self.hamming_threshold,
            "cosine_threshold": self.cosine_threshold,
            "ssim_threshold": self.ssim_threshold,
            "tagger": {
                "name": self.tagger.name,
                "model_path": self.tagger.model_path,
                "tags_csv": self.tagger.tags_csv,
                "thresholds": self.tagger.thresholds,
            },
            "embed_model": {
                "name": self.embed_model.name,
                "pretrained": self.embed_model.pretrained,
                "device": self.embed_model.device,
                "dim": self.embed_model.dim,
            },
            "index_dir": self.index_dir or default_index_dir(),
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        file_path = Path(path)
        payload = self.to_mapping()
        file_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def normalise_roots(values: Sequence[str | os.PathLike[str]]) -> list[str]:
    return [str(Path(value).expanduser()) for value in values]


def normalise_excluded(values: Sequence[str | os.PathLike[str]]) -> list[str]:
    if not values:
        return _default_excluded()
    return [str(Path(value).expanduser()) for value in values]


def _normalise_exts(values: Any, fallback: set[str]) -> set[str]:
    if not values:
        return set(fallback)
    normalised: set[str] = set()
    for value in values:
        ext = PipelineSettings._normalise_ext(str(value))
        if ext:
            normalised.add(ext)
    return normalised or set(fallback)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_optional_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


__all__ = [
    "PipelineSettings",
    "TaggerSettings",
    "EmbedModel",
    "default_index_dir",
    "normalise_roots",
    "normalise_excluded",
]
