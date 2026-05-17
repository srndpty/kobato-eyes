"""Shared ONNX backend helpers for tagger implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    cuda_provider: str = CUDA_PROVIDER,
    cpu_provider: str = CPU_PROVIDER,
) -> ProviderPlan:
    """Return ONNX Runtime provider attempts and caller-loggable diagnostics."""

    available_set = set(available)
    warnings: list[str] = []
    infos: list[str] = []
    if requested is not None:
        requested_list = list(requested)
        if cuda_provider in requested_list and cuda_provider not in available_set:
            warnings.append(f"{cuda_provider} requested but not available; falling back to {cpu_provider}")
            return ProviderPlan(attempts=[[cpu_provider]], warnings=warnings, infos=infos)
        return ProviderPlan(attempts=[requested_list], warnings=warnings, infos=infos)

    if available_set and cuda_provider not in available_set:
        infos.append(f"{cuda_provider} not reported by runtime; CPU provider will be used if CUDA fails")
    return ProviderPlan(attempts=[[cuda_provider], [cpu_provider]], warnings=warnings, infos=infos)


def validate_label_count(output_dim: int, label_count: int, *, backend_name: str) -> None:
    """Raise a clear error when model output dimensions and labels disagree."""

    if output_dim != label_count:
        raise RuntimeError(
            f"{backend_name}: model output dimension {output_dim} does not match label count {label_count}"
        )


def cuda_provider_options(provider_list: Iterable[str]) -> list[dict[str, str]] | None:
    """Return optional ONNX Runtime provider options from environment variables."""

    providers = list(provider_list)
    if CUDA_PROVIDER not in providers:
        return None

    options_by_provider: dict[str, dict[str, str]] = {provider: {} for provider in providers}
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


__all__ = [
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "ProviderPlan",
    "cuda_provider_options",
    "plan_provider_attempts",
    "resolve_existing_file",
    "validate_label_count",
]
