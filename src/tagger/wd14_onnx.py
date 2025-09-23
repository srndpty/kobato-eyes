"""ONNX Runtime implementation of the WD14 tagger."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from tagger.base import (
    ITagger,
    MaxTagsMap,
    TagCategory,
    TagPrediction,
    TagResult,
    ThresholdMap,
)

try:  # pragma: no cover - import is environment dependent
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - graceful degradation
    ort = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class _Label:
    name: str
    category: TagCategory


_CATEGORY_LOOKUP = {
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class WD14Tagger(ITagger):
    """WD14 tagger backed by ONNX Runtime with CUDA acceleration when available."""

    def __init__(
        self,
        model_path: str | Path,
        labels_csv: str | Path,
        *,
        providers: Iterable[str] | None = None,
        input_size: int = 448,
        default_thresholds: ThresholdMap | None = None,
        default_max_tags: MaxTagsMap | None = None,
    ) -> None:
        if ort is None:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "onnxruntime is required to use WD14Tagger"
            ) from _IMPORT_ERROR

        self._model_path = Path(model_path)
        self._labels = self._load_labels(labels_csv)
        self._input_size = int(input_size)
        self._default_thresholds = dict(default_thresholds or {})
        self._default_max_tags = dict(default_max_tags or {})

        provider_list = (
            list(providers)
            if providers is not None
            else [
                "CUDAExecutionProvider",
                "ROCMExecutionProvider",
                "DirectMLExecutionProvider",
                "CPUExecutionProvider",
            ]
        )

        session_options = ort.SessionOptions()
        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=session_options,
            providers=provider_list,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

        if len(self._output_names) != 1:
            raise RuntimeError(
                "Expected a single output tensor from WD14 ONNX model, got "
                f"{self._output_names}"
            )

    @staticmethod
    def _load_labels(labels_csv: str | Path) -> list[_Label]:
        labels: list[_Label] = []
        with Path(labels_csv).open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) == 1:
                    tag_name = row[0].strip()
                    category = TagCategory.GENERAL
                else:
                    tag_name = row[0].strip()
                    category_key = row[1].strip().lower()
                    category = _CATEGORY_LOOKUP.get(category_key, TagCategory.GENERAL)
                if tag_name:
                    labels.append(_Label(name=tag_name, category=category))
        if not labels:
            raise ValueError("No labels parsed from WD14 label CSV")
        return labels

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        resample = getattr(Image, "Resampling", Image).BICUBIC  # type: ignore[attr-defined]
        resized = rgb.resize((self._input_size, self._input_size), resample)
        array = np.asarray(resized).astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, 0)

    @staticmethod
    def _resolve_thresholds(
        defaults: dict[TagCategory, float], overrides: ThresholdMap | None
    ) -> dict[TagCategory, float]:
        merged = dict(defaults)
        if overrides:
            merged.update(overrides)
        return merged

    @staticmethod
    def _resolve_max_tags(
        defaults: dict[TagCategory, int], overrides: MaxTagsMap | None
    ) -> dict[TagCategory, int]:
        merged = dict(defaults)
        if overrides:
            merged.update(overrides)
        return merged

    def infer_batch(
        self,
        images: Iterable[Image.Image],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        image_list = list(images)
        if not image_list:
            return []

        batch = np.vstack([self._preprocess(image) for image in image_list])
        outputs = self._session.run(self._output_names, {self._input_name: batch})
        logits = outputs[0]

        if logits.shape[1] != len(self._labels):
            raise RuntimeError(
                f"Model output dimension {logits.shape[1]} does not match label count {len(self._labels)}"
            )

        probabilities = _sigmoid(logits)
        resolved_thresholds = self._resolve_thresholds(
            self._default_thresholds, thresholds
        )
        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)

        results: list[TagResult] = []
        for probs in probabilities:
            predictions: list[TagPrediction] = []
            by_category: dict[TagCategory, list[TagPrediction]] = {}
            for label, score in zip(self._labels, probs):
                probability = float(score)
                threshold = resolved_thresholds.get(label.category, 0.0)
                if probability < threshold:
                    continue
                prediction = TagPrediction(
                    name=label.name, score=probability, category=label.category
                )
                by_category.setdefault(label.category, []).append(prediction)

            for category, preds in by_category.items():
                preds.sort(key=lambda item: item.score, reverse=True)
                limit = resolved_limits.get(category)
                if limit is not None:
                    preds = preds[: max(limit, 0)]
                predictions.extend(preds)

            predictions.sort(key=lambda item: item.score, reverse=True)
            results.append(TagResult(tags=predictions))
        return results


__all__ = ["WD14Tagger"]
