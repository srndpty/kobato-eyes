"""PixAI ONNX tagger support."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from tagger.base import MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap
from tagger.wd14_onnx import WD14Tagger, _Label

logger = logging.getLogger(__name__)

_CATEGORY_MAP: Mapping[str, TagCategory] = {
    "0": TagCategory.GENERAL,
    "general": TagCategory.GENERAL,
    "1": TagCategory.CHARACTER,
    "character": TagCategory.CHARACTER,
    "2": TagCategory.RATING,
    "rating": TagCategory.RATING,
    "3": TagCategory.COPYRIGHT,
    "copyright": TagCategory.COPYRIGHT,
    "4": TagCategory.ARTIST,
    "artist": TagCategory.ARTIST,
    "5": TagCategory.META,
    "meta": TagCategory.META,
}


def _parse_category(value: str | None) -> TagCategory:
    if not value:
        return TagCategory.GENERAL
    lowered = value.strip().lower()
    if not lowered:
        return TagCategory.GENERAL
    if lowered in _CATEGORY_MAP:
        return _CATEGORY_MAP[lowered]
    try:
        numeric = int(float(lowered))
        return TagCategory(numeric)
    except (ValueError, TypeError):
        return TagCategory.GENERAL


def _parse_count(value: str | None) -> int:
    if not value:
        return 0
    stripped = value.strip()
    if not stripped:
        return 0
    try:
        return int(float(stripped))
    except (ValueError, TypeError):
        return 0


def _parse_ips(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        logger.debug("PixAI: failed to parse ips column %r", value)
        return []
    if not isinstance(parsed, list):
        return []
    result: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
    return result


def merge_character_ips(
    tags: Sequence[TagPrediction],
    ips_lookup: Mapping[str, Sequence[str]],
) -> list[TagPrediction]:
    """Return ``tags`` with copyright predictions merged from ``ips_lookup``."""

    merged: list[TagPrediction] = []
    seen: set[str] = set()
    for tag in tags:
        name = tag.name.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(tag)
        if tag.category != TagCategory.CHARACTER:
            continue
        extras = ips_lookup.get(key)
        if not extras:
            continue
        for ip_name in extras:
            cleaned = ip_name.strip()
            if not cleaned:
                continue
            ip_key = cleaned.lower()
            if ip_key in seen:
                continue
            merged.append(TagPrediction(name=cleaned, score=tag.score, category=TagCategory.COPYRIGHT))
            seen.add(ip_key)
    return merged


class PixaiOnnxTagger(WD14Tagger):
    """WD14-compatible tagger that understands PixAI ``ips`` metadata."""

    def __init__(
        self,
        model_path: str | Path,
        labels_csv: str | Path | None = None,
        *,
        tags_csv: str | Path | None = None,
        providers: Iterable[str] | None = None,
        input_size: int = 448,
        default_thresholds: ThresholdMap | None = None,
        default_max_tags: MaxTagsMap | None = None,
    ) -> None:
        self._pixai_valid_mask: np.ndarray | None = None
        self._pixai_ips: dict[str, list[str]] = {}
        super().__init__(
            model_path,
            labels_csv=labels_csv,
            tags_csv=tags_csv,
            providers=providers,
            input_size=input_size,
            default_thresholds=default_thresholds,
            default_max_tags=default_max_tags,
        )
        mask = self._pixai_valid_mask
        if mask is not None and getattr(self, "_default_thr_vec", None) is not None:
            invalid = ~mask
            if np.any(invalid):
                self._default_thr_vec = self._default_thr_vec.astype(np.float32, copy=True)
                self._default_thr_vec[invalid] = np.inf

    def _load_labels(self, labels_csv: str | Path) -> list[_Label]:  # type: ignore[override]
        path = Path(labels_csv)
        labels: list[_Label] = []
        valid_mask: list[bool] = []
        ips_lookup: dict[str, list[str]] = {}

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if row is None:
                    continue
                cells = [cell.strip() for cell in row]
                if not cells:
                    labels.append(_Label(name="", category=TagCategory.GENERAL, count=0))
                    valid_mask.append(False)
                    continue
                if not any(cells):
                    labels.append(_Label(name="", category=TagCategory.GENERAL, count=0))
                    valid_mask.append(False)
                    continue
                first = cells[0].lower()
                if first.startswith("#"):
                    continue
                if first in {"id", "tag_id", "tagid"}:
                    continue
                if len(cells) > 1 and cells[1].lower() in {"name", "tag", "tag_name"}:
                    continue

                name: str
                category_value: str | None = None
                count_value: str | None = None
                ips_value: str | None = None
                if cells and (cells[0].lstrip("-").isdigit() or cells[0] == "") and len(cells) >= 3:
                    name = cells[2]
                    category_value = cells[3] if len(cells) > 3 else None
                    count_value = cells[4] if len(cells) > 4 else None
                    ips_value = cells[5] if len(cells) > 5 else None
                else:
                    name = cells[0]
                    category_value = cells[1] if len(cells) > 1 else None
                    count_value = cells[2] if len(cells) > 2 else None
                    ips_value = cells[3] if len(cells) > 3 else None

                cleaned_name = name.strip()
                category_enum = _parse_category(category_value)
                count = _parse_count(count_value)
                if not cleaned_name:
                    labels.append(_Label(name="", category=TagCategory.GENERAL, count=count))
                    valid_mask.append(False)
                    continue

                labels.append(_Label(name=cleaned_name, category=category_enum, count=count))
                valid_mask.append(True)
                ips_list = _parse_ips(ips_value)
                if category_enum == TagCategory.CHARACTER and ips_list:
                    ips_lookup[cleaned_name.lower()] = ips_list

        if not labels:
            raise ValueError("No labels parsed from PixAI CSV")

        self._pixai_valid_mask = np.array(valid_mask, dtype=bool) if valid_mask else None
        self._pixai_ips = ips_lookup
        return labels

    def _build_threshold_vector(self, thresholds: dict[TagCategory | int, float]) -> np.ndarray:
        vec = super()._build_threshold_vector(thresholds)
        mask = self._pixai_valid_mask
        if mask is not None and vec.shape[0] == mask.size:
            invalid = ~mask
            if np.any(invalid):
                vec = vec.astype(np.float32, copy=True)
                vec[invalid] = np.inf
        return vec

    def _postprocess_logits_topk(
        self,
        logits: np.ndarray,
        *,
        thresholds: ThresholdMap | None,
        max_tags: MaxTagsMap | None,
    ) -> list[TagResult]:
        mask = self._pixai_valid_mask
        if mask is not None and logits.shape[1] == mask.size:
            logits = np.array(logits, copy=True)
            logits[:, ~mask] = -np.inf
        results = super()._postprocess_logits_topk(logits, thresholds=thresholds, max_tags=max_tags)
        if not results:
            return results
        if not self._pixai_ips:
            for result in results:
                result.tags = merge_character_ips(result.tags, {})
            return results
        lookup = self._pixai_ips
        for result in results:
            result.tags = merge_character_ips(result.tags, lookup)
        return results


__all__ = ["PixaiOnnxTagger", "merge_character_ips"]

