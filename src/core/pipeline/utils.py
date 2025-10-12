from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from core.config import PipelineSettings
from tagger.base import TagCategory
from tagger.labels_util import discover_labels_csv, load_selected_tags

# ---- formatting / digest helpers ------------------------------------------------


def _format_sig_mapping(mapping: Mapping[str, float | int]) -> str:
    if not mapping:
        return "none"
    parts: list[str] = []
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, float):
            formatted = format(value, ".6f").rstrip("0").rstrip(".")
        else:
            formatted = str(int(value))
        parts.append(f"{key}={formatted}")
    return ",".join(parts)


def _normalise_sig_source(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        candidate = value
    else:
        try:
            candidate = Path(str(value))
        except (TypeError, ValueError):
            return str(value)
    expanded = candidate.expanduser()
    try:
        resolved = expanded.resolve(strict=False)
    except OSError:
        resolved = expanded.absolute()
    return str(resolved)


def _digest_identifier(value: str | Path | None) -> str:
    normalised = _normalise_sig_source(value)
    if normalised is None:
        return "none"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---- serialisers used by resolver / pipeline -----------------------------------


def _serialise_thresholds(
    thresholds: Mapping[TagCategory, float] | None,
) -> dict[str, float]:
    if not thresholds:
        return {}
    return {category.name.lower(): float(value) for category, value in thresholds.items()}


def _serialise_max_tags(
    max_tags: Mapping[TagCategory, int] | None,
) -> dict[str, int]:
    if not max_tags:
        return {}
    return {category.name.lower(): int(value) for category, value in max_tags.items()}


def detect_tagger_provider(settings: PipelineSettings) -> str:
    """Return the effective tagger provider after considering auto-detection."""

    configured = str(getattr(settings.tagger, "provider", "auto") or "auto").lower()
    if configured in {"wd14", "pixai"}:
        return configured
    if configured not in {"", "auto"}:
        return "wd14"

    csv_candidate = discover_labels_csv(settings.tagger.model_path, settings.tagger.tags_csv)
    if csv_candidate is None:
        return "wd14"
    try:
        tags = load_selected_tags(csv_candidate)
    except Exception:
        return "wd14"
    return "pixai" if any(tag.ips for tag in tags) else "wd14"


__all__ = [
    "_format_sig_mapping",
    "_normalise_sig_source",
    "_digest_identifier",
    "_serialise_thresholds",
    "_serialise_max_tags",
    "detect_tagger_provider",
]
