from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Mapping

from core.config import PipelineSettings
from tagger.base import TagCategory
from tagger.labels_util import discover_labels_csv, load_selected_tags

# 既定値（必要に応じて調整）
WD14_DEFAULT_THRESHOLDS = {
    TagCategory.GENERAL: 0.35,
    TagCategory.CHARACTER: 0.25,
    TagCategory.COPYRIGHT: 0.25,
}
PIXAI_DEFAULT_THRESHOLDS = {
    TagCategory.GENERAL: 0.90,
    TagCategory.CHARACTER: 0.95,
    TagCategory.COPYRIGHT: 0.95,
}
PIXAI_DEFAULT_MAX_TAGS = {
    TagCategory.GENERAL: 50,
    TagCategory.CHARACTER: 3,
    TagCategory.COPYRIGHT: 3,
}


def provider_default_thresholds(provider: str) -> dict[TagCategory, float]:
    return PIXAI_DEFAULT_THRESHOLDS.copy() if provider == "pixai" else WD14_DEFAULT_THRESHOLDS.copy()


def provider_default_max_tags(provider: str) -> dict[TagCategory, int]:
    return PIXAI_DEFAULT_MAX_TAGS.copy() if provider == "pixai" else {}


def is_wd14_defaults(th: dict[TagCategory, float] | None) -> bool:
    if not th:
        return False
    try:
        return all(abs(th.get(k, 0.0) - v) < 1e-6 for k, v in WD14_DEFAULT_THRESHOLDS.items())
    except Exception:
        return False


def overlay_defaults(user_map: dict, defaults_map: dict):
    """user指定を優先し、欠けているカテゴリだけdefaultsで埋める"""
    result = defaults_map.copy()
    result.update(user_map or {})
    return result


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


logger = logging.getLogger(__name__)


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
        logger.info(
            "Loaded %d tags from %s (first=%r, last=%r)",
            len(tags),
            csv_candidate,
            tags[0].name if tags else None,
            tags[-1].name if tags else None,
        )
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


# --- provider defaults ----------------------------------------------------------
# PixAI はロジットが高めに出やすいので、保守的な初期値を返す
# def provider_default_thresholds(provider: str) -> Mapping[TagCategory, float] | None:
#     if provider == "pixai":
#         return {
#             TagCategory.GENERAL: 0.90,  # logit ≈ 2.197
#             TagCategory.CHARACTER: 0.95,  # logit ≈ 2.944
#             TagCategory.COPYRIGHT: 0.95,  # logit ≈ 2.944
#         }
#     return None


# def provider_default_max_tags(provider: str) -> Mapping[TagCategory, int] | None:
#     if provider == "pixai":
#         return {
#             TagCategory.GENERAL: 50,
#             TagCategory.CHARACTER: 3,
#             TagCategory.COPYRIGHT: 3,
#         }
#     return None
