"""Opt-in CUDA test for a downloaded PixAI v1 model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tagger.pixai_v1_onnx import PixaiV1OnnxTagger

pytestmark = pytest.mark.gpu


def test_pixai_v1_cuda_real_model() -> None:
    """Run two images through fixed batching and verify probabilities/categories."""
    model = os.getenv("KE_PIXAI_V1_MODEL")
    if not model:
        pytest.skip("Set KE_PIXAI_V1_MODEL to a downloaded model.onnx")
    assert Path(model).is_file()
    tagger = PixaiV1OnnxTagger(model, providers=["CUDAExecutionProvider"])
    try:
        assert "CUDAExecutionProvider" in tagger._session.get_providers()
        images = [np.full((1008, 1008, 3), value, np.uint8) for value in [0, 255]]
        results = tagger.infer_batch_prepared(tagger.prepare_batch_from_rgb_np(images))
        assert len(results) == 2
        assert all(
            0 <= tag.score <= 1 and int(tag.category) in {0, 2, 3, 4, 5, 6} for result in results for tag in result.tags
        )
    finally:
        tagger.close()
