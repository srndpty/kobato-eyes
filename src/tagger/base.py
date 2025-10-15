"""Tagging abstractions used across kobato-eyes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image


class TagCategory(IntEnum):
    """Logical Danbooru-style tag categories."""

    GENERAL = 0
    ARTIST = 1
    RATING = 2
    COPYRIGHT = 3
    CHARACTER = 4
    META = 5


@dataclass(frozen=True)
class TagPrediction:
    """Single tag prediction returned from a tagger."""

    name: str
    score: float
    category: TagCategory


@dataclass
class TagResult:
    """Aggregated predictions for a single image."""

    tags: list[TagPrediction]


ThresholdMap = Mapping[TagCategory, float]
MaxTagsMap = Mapping[TagCategory, int]


@runtime_checkable
class ITagger(Protocol):
    """Interface all tagging implementations must satisfy."""

    def prepare_batch_from_rgb_np(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Convert raw RGB image arrays into a model-ready float32 batch."""

    def infer_batch_prepared(
        self,
        batch: np.ndarray,
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        """Run inference over a preprocessed batch produced by ``prepare_batch_from_rgb_np``."""

    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        """Run inference over a batch of images."""


__all__ = [
    "ITagger",
    "TagCategory",
    "TagPrediction",
    "TagResult",
    "ThresholdMap",
    "MaxTagsMap",
]
