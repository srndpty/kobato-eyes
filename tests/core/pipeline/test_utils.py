"""Tests for pipeline utility helpers."""

from __future__ import annotations

import core.pipeline.utils as pipeline_utils
from core.config import PipelineSettings


def test_detect_tagger_provider_prefers_model_output_detection(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """PixAI ONNX outputs should override CSV-only auto detection."""

    settings = PipelineSettings.from_mapping(
        {
            "tagger": {
                "name": "wd14-onnx",
                "model_path": "model.onnx",
                "provider": "auto",
            }
        }
    )
    monkeypatch.setattr(pipeline_utils, "detect_provider_from_model_outputs", lambda _path, label_count: "pixai")

    assert pipeline_utils.detect_tagger_provider(settings) == "pixai"
