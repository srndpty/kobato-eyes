"""Shared ONNX backend helpers for tagger implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TENSORRT_PROVIDER = "TensorrtExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class ProviderPlan:
    """Provider attempts plus diagnostics for ONNX Runtime session creation."""

    attempts: list[list[str]]
    warnings: list[str]
    infos: list[str]


def resolve_existing_file(path: str | Path, *, label: str) -> Path:
    """Return an existing file path or raise a user-facing error."""

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found at {resolved}")
    return resolved


def plan_provider_attempts(
    requested: Iterable[str] | None,
    available: Iterable[str],
    *,
    tensorrt_provider: str = TENSORRT_PROVIDER,
    cuda_provider: str = CUDA_PROVIDER,
    cpu_provider: str = CPU_PROVIDER,
) -> ProviderPlan:
    """Return ONNX Runtime provider attempts and caller-loggable diagnostics."""

    available_set = set(available)
    warnings: list[str] = []
    infos: list[str] = []
    if requested is not None:
        requested_list = list(requested)
        if tensorrt_provider in requested_list and tensorrt_provider not in available_set:
            warnings.append(f"{tensorrt_provider} requested but not available; falling back to {cuda_provider}")
            if cuda_provider in available_set:
                return ProviderPlan(attempts=[[cuda_provider], [cpu_provider]], warnings=warnings, infos=infos)
            return ProviderPlan(attempts=[[cpu_provider]], warnings=warnings, infos=infos)
        if tensorrt_provider in requested_list:
            return ProviderPlan(
                attempts=[requested_list, [cuda_provider], [cpu_provider]],
                warnings=warnings,
                infos=infos,
            )
        if cuda_provider in requested_list and cuda_provider not in available_set:
            warnings.append(f"{cuda_provider} requested but not available; falling back to {cpu_provider}")
            return ProviderPlan(attempts=[[cpu_provider]], warnings=warnings, infos=infos)
        return ProviderPlan(attempts=[requested_list], warnings=warnings, infos=infos)

    if os.getenv("KE_ORT_TENSORRT", "0") == "1":
        if tensorrt_provider in available_set:
            infos.append(f"{tensorrt_provider} requested by KE_ORT_TENSORRT=1")
            return ProviderPlan(
                attempts=[[tensorrt_provider, cuda_provider, cpu_provider], [cuda_provider], [cpu_provider]],
                warnings=warnings,
                infos=infos,
            )
        warnings.append(f"{tensorrt_provider} requested by KE_ORT_TENSORRT=1 but not available")

    if available_set and cuda_provider not in available_set:
        infos.append(f"{cuda_provider} not reported by runtime; CPU provider will be used if CUDA fails")
    return ProviderPlan(attempts=[[cuda_provider], [cpu_provider]], warnings=warnings, infos=infos)


def validate_label_count(output_dim: int, label_count: int, *, backend_name: str) -> None:
    """Raise a clear error when model output dimensions and labels disagree."""

    if output_dim != label_count:
        raise RuntimeError(
            f"{backend_name}: model output dimension {output_dim} does not match label count {label_count}"
        )


def onnx_provider_options(provider_list: Iterable[str]) -> list[dict[str, str]] | None:
    """Return optional ONNX Runtime provider options from environment variables."""

    providers = list(provider_list)
    if CUDA_PROVIDER not in providers and TENSORRT_PROVIDER not in providers:
        return None

    options_by_provider: dict[str, dict[str, str]] = {provider: {} for provider in providers}
    tensorrt_options = options_by_provider.get(TENSORRT_PROVIDER)
    if tensorrt_options is not None:
        engine_cache_path = os.getenv("KE_ORT_TENSORRT_CACHE_PATH")
        if engine_cache_path:
            tensorrt_options["trt_engine_cache_enable"] = "True"
            tensorrt_options["trt_engine_cache_path"] = engine_cache_path

        timing_cache_path = os.getenv("KE_ORT_TENSORRT_TIMING_CACHE_PATH")
        if timing_cache_path:
            tensorrt_options["trt_timing_cache_enable"] = "True"
            tensorrt_options["trt_timing_cache_path"] = timing_cache_path

        if os.getenv("KE_ORT_TENSORRT_FP16", "0") == "1":
            tensorrt_options["trt_fp16_enable"] = "True"

        max_workspace_mb = os.getenv("KE_ORT_TENSORRT_MAX_WORKSPACE_MB")
        if max_workspace_mb:
            try:
                limit_bytes = max(1, int(max_workspace_mb)) * 1024 * 1024
            except ValueError:
                limit_bytes = 0
            if limit_bytes > 0:
                tensorrt_options["trt_max_workspace_size"] = str(limit_bytes)

    cuda_options = options_by_provider.get(CUDA_PROVIDER, {})
    mem_limit_mb = os.getenv("KE_ORT_CUDA_MEM_LIMIT_MB")
    if mem_limit_mb:
        try:
            limit_bytes = max(1, int(mem_limit_mb)) * 1024 * 1024
        except ValueError:
            limit_bytes = 0
        if limit_bytes > 0:
            cuda_options["gpu_mem_limit"] = str(limit_bytes)

    cudnn_search = os.getenv("KE_ORT_CUDA_CUDNN_CONV_ALGO_SEARCH")
    if cudnn_search:
        cuda_options["cudnn_conv_algo_search"] = cudnn_search

    if not any(options_by_provider.values()):
        return None
    return [options_by_provider.get(provider, {}) for provider in providers]


def cuda_provider_options(provider_list: Iterable[str]) -> list[dict[str, str]] | None:
    """Backward-compatible alias for ONNX Runtime provider options."""

    return onnx_provider_options(provider_list)


__all__ = [
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "ProviderPlan",
    "TENSORRT_PROVIDER",
    "cuda_provider_options",
    "onnx_provider_options",
    "plan_provider_attempts",
    "resolve_existing_file",
    "validate_label_count",
]
