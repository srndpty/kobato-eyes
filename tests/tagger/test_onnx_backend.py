"""Tests for shared ONNX backend helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tagger.onnx_backend import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    TENSORRT_PROVIDER,
    onnx_provider_options,
    plan_provider_attempts,
    resolve_existing_file,
    validate_label_count,
)


def test_plan_provider_attempts_explicit_cuda_missing_falls_back_to_cpu() -> None:
    plan = plan_provider_attempts([CUDA_PROVIDER], [CPU_PROVIDER])

    assert plan.attempts == [[CPU_PROVIDER]]
    assert plan.warnings == [f"{CUDA_PROVIDER} requested but not available; falling back to {CPU_PROVIDER}"]
    assert plan.infos == []


def test_plan_provider_attempts_auto_records_cpu_fallback_hint() -> None:
    plan = plan_provider_attempts(None, [CPU_PROVIDER])

    assert plan.attempts == [[CUDA_PROVIDER], [CPU_PROVIDER]]
    assert plan.warnings == []
    assert plan.infos == [f"{CUDA_PROVIDER} not reported by runtime; CPU provider will be used if CUDA fails"]


def test_plan_provider_attempts_explicit_tensorrt_uses_cuda_fallback() -> None:
    plan = plan_provider_attempts(
        [TENSORRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER],
        [TENSORRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER],
    )

    assert plan.attempts == [[TENSORRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER], [CUDA_PROVIDER], [CPU_PROVIDER]]
    assert plan.warnings == []
    assert plan.infos == []


def test_plan_provider_attempts_tensorrt_missing_falls_back_to_cuda() -> None:
    plan = plan_provider_attempts([TENSORRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER], [CUDA_PROVIDER, CPU_PROVIDER])

    assert plan.attempts == [[CUDA_PROVIDER], [CPU_PROVIDER]]
    assert plan.warnings == [f"{TENSORRT_PROVIDER} requested but not available; falling back to {CUDA_PROVIDER}"]
    assert plan.infos == []


def test_onnx_provider_options_includes_tensorrt_cache_and_fp16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "trt-cache"
    monkeypatch.setenv("KE_ORT_TENSORRT_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("KE_ORT_TENSORRT_FP16", "1")

    options = onnx_provider_options([TENSORRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER])

    assert options is not None
    assert options[0]["trt_engine_cache_enable"] == "True"
    assert options[0]["trt_engine_cache_path"] == str(cache_path)
    assert options[0]["trt_fp16_enable"] == "True"


def test_resolve_existing_file_reports_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.onnx"

    with pytest.raises(FileNotFoundError, match="WD14 model not found"):
        resolve_existing_file(missing, label="WD14 model")


def test_validate_label_count_reports_model_label_mismatch() -> None:
    with pytest.raises(RuntimeError, match="PixAI: model output dimension 3 does not match label count 2"):
        validate_label_count(3, 2, backend_name="PixAI")
