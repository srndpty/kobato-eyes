"""ONNX Runtime implementation of the Pixai tagger."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from tagger import labels_util
from tagger.base import MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap

from . import wd14_onnx

logger = logging.getLogger(__name__)


def _parse_ips_cell(raw: str) -> tuple[str, ...]:
    """Parse the ``ips`` column from the Pixai ``selected_tags.csv``."""

    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Pixai: failed to decode ips field %s", raw)
        return ()
    if not isinstance(parsed, (list, tuple)):
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in parsed:
        text = str(entry).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


class PixaiOnnxTagger(wd14_onnx.WD14Tagger):
    """Pixai tagger backed by ONNX Runtime with automatic copyright merging."""

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
        self._pixai_copyright_lookup: dict[tuple[str, TagCategory], tuple[str, ...]] = {}
        super().__init__(
            model_path,
            labels_csv,
            tags_csv=tags_csv,
            providers=providers,
            input_size=input_size,
            default_thresholds=default_thresholds,
            default_max_tags=default_max_tags,
        )
        self._update_input_size_from_session()

    def _load_labels(self, labels_csv: str | Path) -> list[wd14_onnx._Label]:  # type: ignore[name-defined]
        base_labels = wd14_onnx.WD14Tagger._load_labels(labels_csv)
        extras = self._load_pixai_metadata(labels_csv)
        if len(extras) != len(base_labels):
            logger.warning(
                "Pixai: labels count %d does not match ips metadata %d", len(base_labels), len(extras)
            )
        if len(extras) < len(base_labels):
            extras.extend([()] * (len(base_labels) - len(extras)))
        elif len(extras) > len(base_labels):
            extras = extras[: len(base_labels)]
        lookup: dict[tuple[str, TagCategory], tuple[str, ...]] = {}
        for index, label in enumerate(base_labels):
            ips = extras[index]
            if not ips:
                continue
            lookup[(label.name, label.category)] = ips
        self._pixai_copyright_lookup = lookup
        return base_labels

    def infer_batch_prepared(
        self,
        batch_bgr_or_rgb_prepared: np.ndarray,
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        results = super().infer_batch_prepared(
            batch_bgr_or_rgb_prepared,
            thresholds=thresholds,
            max_tags=max_tags,
        )
        self._merge_copyright_tags(results)
        return results

    def infer_batch(
        self,
        images: Iterable[Image.Image],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        results = super().infer_batch(
            images,
            thresholds=thresholds,
            max_tags=max_tags,
        )
        self._merge_copyright_tags(results)
        return results

    def _merge_copyright_tags(self, results: Sequence[TagResult]) -> None:
        if not self._pixai_copyright_lookup:
            return
        for result in results:
            if not result.tags:
                continue
            original = list(result.tags)
            copyright_scores: dict[str, float] = {}
            copyright_order: list[str] = []
            for prediction in original:
                if prediction.category is not TagCategory.COPYRIGHT:
                    continue
                if prediction.name not in copyright_scores or prediction.score > copyright_scores[prediction.name]:
                    copyright_scores[prediction.name] = prediction.score
                if prediction.name not in copyright_order:
                    copyright_order.append(prediction.name)
            for prediction in original:
                if prediction.category is not TagCategory.CHARACTER:
                    continue
                key = (prediction.name, TagCategory.CHARACTER)
                ips = self._pixai_copyright_lookup.get(key)
                if not ips:
                    continue
                for name in ips:
                    if not name:
                        continue
                    score = prediction.score
                    if name not in copyright_scores or score > copyright_scores[name]:
                        copyright_scores[name] = score
                    if name not in copyright_order:
                        copyright_order.append(name)
            merged: list[TagPrediction] = [
                prediction for prediction in original if prediction.category is not TagCategory.COPYRIGHT
            ]
            if copyright_scores:
                merged.extend(
                    TagPrediction(name=name, score=copyright_scores[name], category=TagCategory.COPYRIGHT)
                    for name in copyright_order
                )
            result.tags = merged

    def _load_pixai_metadata(self, labels_csv: str | Path) -> list[tuple[str, ...]]:
        path = Path(labels_csv)
        extras: list[tuple[str, ...]] = []
        for cells in labels_util._iter_csv_rows(path):  # type: ignore[attr-defined]
            tag = labels_util._parse_row(cells)  # type: ignore[attr-defined]
            if tag is None:
                continue
            raw_ips = cells[5] if len(cells) > 5 else ""
            extras.append(_parse_ips_cell(raw_ips))
        return extras

    def _update_input_size_from_session(self) -> None:
        session = getattr(self, "_session", None)
        if session is None:
            return
        try:
            shape = session.get_inputs()[0].shape
        except Exception:
            return
        if not shape or len(shape) < 3:
            return
        height = self._coerce_positive_dim(shape[1])
        width = self._coerce_positive_dim(shape[2]) if len(shape) > 2 else None
        size = height or width
        if size is not None:
            self._input_size = size

    @staticmethod
    def _coerce_positive_dim(value: object) -> int | None:
        if value in (None, -1, "-1"):
            return None
        if isinstance(value, (int, np.integer)) and value > 0:
            return int(value)
        try:
            numeric = int(str(value))
        except (TypeError, ValueError):
            return None
        return numeric if numeric > 0 else None


__all__ = ["PixaiOnnxTagger"]
