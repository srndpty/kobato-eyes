from __future__ import annotations

from pathlib import Path
from typing import Mapping

from core.settings import PipelineSettings
from tagger.base import TagCategory

from .utils import _digest_identifier, _format_sig_mapping

# 外部から使う：_build_threshold_map / _build_max_tags_map / current_tagger_sig

_CATEGORY_KEY_LOOKUP = {
    "general": TagCategory.GENERAL,
    "character": TagCategory.CHARACTER,
    "copyright": TagCategory.COPYRIGHT,
    "artist": TagCategory.ARTIST,
    "meta": TagCategory.META,
    "rating": TagCategory.RATING,
}


def _build_threshold_map(thresholds: dict[str, float]) -> dict[TagCategory, float]:
    mapping: dict[TagCategory, float] = {}
    for key, value in thresholds.items():
        category = _CATEGORY_KEY_LOOKUP.get(key.lower())
        if category is not None:
            mapping[category] = float(value)
    return mapping


def _build_max_tags_map(max_tags: Mapping[str, int] | None) -> dict[TagCategory, int]:
    mapping: dict[TagCategory, int] = {}
    if not max_tags:
        return mapping
    for key, value in max_tags.items():
        category = _CATEGORY_KEY_LOOKUP.get(str(key).lower())
        if category is None:
            continue
        try:
            mapping[category] = int(value)
        except (TypeError, ValueError):
            continue
    return mapping


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


def current_tagger_sig(
    settings: PipelineSettings,
    *,
    thresholds: Mapping[TagCategory, float] | None = None,
    max_tags: Mapping[TagCategory, int] | None = None,
) -> str:
    """アクティブなタガー構成の署名（しきい値/最大数/モデル/CSV）を安定化した文字列で返す。"""
    threshold_map = thresholds or _build_threshold_map(settings.tagger.thresholds)
    max_tags_map = max_tags or _build_max_tags_map(getattr(settings.tagger, "max_tags", None))

    serialised_thresholds = {k.name.lower(): float(v) for k, v in threshold_map.items()}
    serialised_max_tags = {k.name.lower(): int(v) for k, v in max_tags_map.items()}

    tagger_name = str(getattr(settings.tagger, "name", "") or "").lower()
    model_path = getattr(settings.tagger, "model_path", None)
    tags_csv = getattr(settings.tagger, "tags_csv", None)

    model_digest = _digest_identifier(_normalise_sig_source(model_path))
    csv_digest = _digest_identifier(_normalise_sig_source(tags_csv))
    thresholds_part = _format_sig_mapping(serialised_thresholds)
    max_tags_part = _format_sig_mapping(serialised_max_tags)
    return f"{tagger_name}:{model_digest}:csv={csv_digest}:thr={thresholds_part}:max={max_tags_part}"


__all__ = [
    "current_tagger_sig",
    "_build_threshold_map",
    "_build_max_tags_map",
    "_CATEGORY_KEY_LOOKUP",
]
