"""Dummy tagger returning a fixed tag prediction."""

from __future__ import annotations

from typing import Mapping, Sequence

from PIL import Image

from tagger.base import ITagger, TagCategory, TagPrediction, TagResult


class DummyTagger(ITagger):
    """Simple tagger used for tests and offline modes."""

    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds: Mapping[str, float] | None = None,
        max_tags: int | None = None,
    ) -> list[TagResult]:
        predictions = [TagPrediction(name="1girl", score=0.9, category=TagCategory.GENERAL)]
        return [TagResult(tags=predictions)] * len(images)


__all__ = ["DummyTagger"]
