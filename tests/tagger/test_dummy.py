"""Tests for the dummy tagger implementation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from tagger.base import TagCategory
from tagger.dummy import DummyTagger


def test_prepare_batch_from_rgb_np_handles_empty_grayscale_and_rgb() -> None:
    tagger = DummyTagger()

    empty = tagger.prepare_batch_from_rgb_np([])
    assert empty.shape == (0, 0, 0, 3)
    assert empty.dtype == np.float32

    grayscale = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    batch = tagger.prepare_batch_from_rgb_np([grayscale, rgb])

    assert batch.shape == (2, 2, 2, 3)
    assert batch.dtype == np.float32
    assert np.array_equal(batch[0, :, :, 0], grayscale.astype(np.float32))


def test_infer_batch_returns_default_prediction_per_image() -> None:
    tagger = DummyTagger()
    images = [Image.new("L", (2, 2), color=128), Image.new("RGB", (2, 2), color=(1, 2, 3))]

    results = tagger.infer_batch(images)

    assert len(results) == 2
    assert results[0].tags[0].name == "1girl"
    assert results[0].tags[0].score == 0.9
    assert results[0].tags[0].category is TagCategory.GENERAL
