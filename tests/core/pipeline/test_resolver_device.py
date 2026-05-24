"""Tests for tagger resolver execution-device planning."""

from __future__ import annotations

from core.pipeline.resolver import _onnx_providers_for_device


def test_onnx_providers_for_device_maps_user_setting() -> None:
    assert _onnx_providers_for_device("auto") is None
    assert _onnx_providers_for_device(None) is None
    assert _onnx_providers_for_device("cuda") == ["CUDAExecutionProvider"]
    assert _onnx_providers_for_device("cpu") == ["CPUExecutionProvider"]
