"""Dummy tagger returning a fixed tag prediction."""

from __future__ import annotations

from typing import Sequence

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


class DummyTagger(ITagger):
    """Simple tagger used for tests and offline modes."""

    _DEFAULT_TAGS = [TagPrediction(name="1girl", score=0.9, category=TagCategory.GENERAL)]

    def prepare_batch_from_rgb_np(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Convert incoming RGB arrays into a contiguous float32 batch."""

        converted: list[np.ndarray] = []
        for img in images:
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            converted.append(arr.astype(np.float32, copy=False))
        if not converted:
            return np.empty((0, 0, 0, 3), dtype=np.float32)
        return np.stack(converted, axis=0)

    def infer_batch_prepared(
        self,
        batch: np.ndarray,
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        """Return a fixed tag prediction for each prepared image."""

        return [TagResult(tags=list(self._DEFAULT_TAGS)) for _ in range(len(batch))]

    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        rgb_arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) for image in images]
        batch = self.prepare_batch_from_rgb_np(rgb_arrays)
        return self.infer_batch_prepared(batch, thresholds=thresholds, max_tags=max_tags)


__all__ = ["DummyTagger"]
