"""Pydantic schemas for pipeline configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
}
DEFAULT_TAG_THRESHOLDS = {
    "general": 0.35,
    "character": 0.25,
    "copyright": 0.25,
}


def _normalise_path(value: str | Path) -> str:
    return str(Path(value).expanduser())


def _normalise_ext(value: str) -> str:
    ext = value.strip().lower()
    if not ext:
        return ""
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _default_excluded() -> list[str]:
    return [str(Path(path)) for path in DEFAULT_EXCLUDED]


def _default_allow_exts() -> set[str]:
    return set(DEFAULT_ALLOW_EXTS)


def _default_thresholds() -> dict[str, float]:
    return dict(DEFAULT_TAG_THRESHOLDS)


class TaggerSettings(BaseModel):
    """Settings used to configure the tagging pipeline."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = "dummy"
    model_path: str | None = None
    tags_csv: str | None = None
    thresholds: dict[str, float] = Field(default_factory=_default_thresholds)

    @field_validator("model_path", "tags_csv", mode="before")
    @classmethod
    def _validate_optional_path(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return _normalise_path(str(value))

    @field_validator("thresholds", mode="before")
    @classmethod
    def _validate_thresholds(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, Mapping):
            return _default_thresholds()
        thresholds: dict[str, float] = _default_thresholds()
        for key, candidate in value.items():
            try:
                thresholds[str(key)] = float(candidate)
            except (TypeError, ValueError):
                continue
        return thresholds


class PipelineSettings(BaseModel):
    """Validated configuration used to run the processing pipeline."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    roots: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=_default_excluded)
    allow_exts: set[str] = Field(default_factory=_default_allow_exts)
    batch_size: int = 8
    hamming_threshold: int = 10
    ssim_threshold: float = 0.92
    tagger: TaggerSettings = Field(default_factory=TaggerSettings)
    index_dir: str | None = None
    skip_fts_during_write: bool = False

    @model_validator(mode="before")
    @classmethod
    def _prepare_data(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            prepared = dict(data)
            if "excluded" not in prepared and "excludes" in prepared:
                prepared["excluded"] = prepared["excludes"]
            return prepared
        return data

    @field_validator("roots", mode="before")
    @classmethod
    def _normalise_roots(cls, value: Any) -> list[str]:
        if not value:
            return []
        return [_normalise_path(str(item)) for item in value if item]

    @field_validator("excluded", mode="before")
    @classmethod
    def _normalise_excluded(cls, value: Any) -> list[str]:
        if not value:
            return _default_excluded()
        normalised = [_normalise_path(str(item)) for item in value if item]
        return normalised or _default_excluded()

    @field_validator("allow_exts", mode="before")
    @classmethod
    def _normalise_allow_exts(cls, value: Any) -> set[str]:
        if not value:
            return _default_allow_exts()
        normalised: set[str] = set()
        for item in value:
            ext = _normalise_ext(str(item))
            if ext:
                normalised.add(ext)
        return normalised or _default_allow_exts()

    @field_validator("batch_size", mode="before")
    @classmethod
    def _coerce_batch_size(cls, value: Any) -> int:
        try:
            batch_size = int(value)
        except (TypeError, ValueError):
            return 8
        return max(1, batch_size)

    @field_validator("hamming_threshold", mode="before")
    @classmethod
    def _coerce_hamming(cls, value: Any) -> int:
        try:
            threshold = int(value)
        except (TypeError, ValueError):
            return 10
        return max(0, threshold)

    @field_validator("ssim_threshold", mode="before")
    @classmethod
    def _coerce_ssim(cls, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.92

    @field_validator("index_dir", mode="before")
    @classmethod
    def _normalise_index_dir(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return _normalise_path(str(value))

    def resolved_index_dir(self, default: str) -> str:
        """Return the configured index directory, using ``default`` when unset."""

        return self.index_dir or default

    def to_mapping(self, *, default_index_dir: str | None = None) -> dict[str, Any]:
        """Return a serialisable representation of the configuration."""

        payload = self.model_dump()
        payload["roots"] = [str(Path(path)) for path in self.roots]
        payload["excluded"] = [str(Path(path)) for path in self.excluded]
        payload["allow_exts"] = sorted(self.allow_exts)
        payload["index_dir"] = self.index_dir or default_index_dir
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PipelineSettings":
        if not isinstance(data, Mapping):
            data = {}
        return cls.model_validate(data)


__all__ = [
    "DEFAULT_ALLOW_EXTS",
    "DEFAULT_EXCLUDED",
    "DEFAULT_TAG_THRESHOLDS",
    "PipelineSettings",
    "TaggerSettings",
]
