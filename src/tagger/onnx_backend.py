"""Shared ONNX backend helpers for tagger implementations."""

from __future__ import annotations

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


__all__ = [
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "ProviderPlan",
    "plan_provider_attempts",
    "resolve_existing_file",
    "validate_label_count",
]
