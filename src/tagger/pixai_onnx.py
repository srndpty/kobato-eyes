"""PixAI ONNX tagger implementation sharing logic with WD14."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tagger.base import TagCategory, TagPrediction, TagResult
from tagger.wd14_onnx import WD14Tagger, _Label

logger = logging.getLogger(__name__)


class PixaiOnnxTagger(WD14Tagger):
    """PixAI tagger compatible with the WD14 ONNX runtime integration."""

    def __init__(
        self,
        model_path: str | Path,
        labels_csv: str | Path | None = None,
        *,
        tags_csv: str | Path | None = None,
        providers: Iterable[str] | None = None,
        input_size: int = 448,
        default_thresholds: dict[TagCategory, float] | None = None,
        default_max_tags: dict[TagCategory, int] | None = None,
    ) -> None:
        super().__init__(
            model_path,
            labels_csv=labels_csv,
            tags_csv=tags_csv,
            providers=providers,
            input_size=input_size,
            default_thresholds=default_thresholds,
            default_max_tags=default_max_tags,
        )
        self._character_ips: dict[str, tuple[str, ...]] = {
            label.name: tuple(label.ips)
            for label in self._labels
            if getattr(label, "ips", ()) and label.name and label.category == TagCategory.CHARACTER
        }

    @staticmethod
    def _parse_category(value: Any) -> TagCategory:
        try:
            numeric = int(float(str(value).strip()))
        except (TypeError, ValueError):
            numeric = 0
        try:
            return TagCategory(numeric)
        except ValueError:
            return TagCategory.GENERAL

    @staticmethod
    def _parse_count(value: Any) -> int:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _parse_ips(value: str | None) -> tuple[str, ...]:
        if not value:
            return ()
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("PixAI: failed to parse ips field '%s'", text)
            return ()
        if not isinstance(parsed, Sequence):
            return ()
        cleaned: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
        return tuple(cleaned)

    @staticmethod
    def _load_labels(labels_csv: str | Path) -> list[_Label]:  # type: ignore[override]
        path = Path(labels_csv)
        labels: list[_Label] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header_skipped = False
            for row in reader:
                if not row:
                    labels.append(_Label(name="", category=TagCategory.GENERAL))
                    continue
                cells = [cell.strip() for cell in row]
                if not header_skipped and cells and cells[0].lower() in {"id", "tag_id"}:
                    header_skipped = True
                    continue
                padded = (cells + ["", "", "", "", ""])[:6]
                name_cell = padded[2]
                category_cell = padded[3]
                count_cell = padded[4]
                ips_cell = padded[5]
                category = PixaiOnnxTagger._parse_category(category_cell)
                count = PixaiOnnxTagger._parse_count(count_cell)
                ips = PixaiOnnxTagger._parse_ips(ips_cell)
                labels.append(
                    _Label(
                        name=name_cell.strip(),
                        category=category,
                        count=count,
                        ips=ips,
                    )
                )
        if not labels:
            raise ValueError("No labels parsed from PixAI label CSV")
        return labels

    def _postprocess_logits_topk(  # type: ignore[override]
        self,
        logits: np.ndarray,
        *,
        thresholds: dict[TagCategory, float] | None,
        max_tags: dict[TagCategory, int] | None,
    ) -> list[TagResult]:
        results = super()._postprocess_logits_topk(logits, thresholds=thresholds, max_tags=max_tags)
        if not self._character_ips:
            return results
        for result in results:
            if not result.tags:
                continue
            existing = {tag.name for tag in result.tags}
            extras: list[TagPrediction] = []
            for tag in result.tags:
                if tag.category != TagCategory.CHARACTER:
                    continue
                ips = self._character_ips.get(tag.name)
                if not ips:
                    continue
                for ip in ips:
                    if ip in existing:
                        continue
                    extras.append(TagPrediction(name=ip, score=tag.score, category=TagCategory.COPYRIGHT))
                    existing.add(ip)
            if extras:
                result.tags.extend(extras)
        return results


__all__ = ["PixaiOnnxTagger"]
