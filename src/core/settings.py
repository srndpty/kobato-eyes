"""Shared settings structures for kobato-eyes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PipelineSettings:
    roots: List[Path] = field(default_factory=list)
    excluded: List[Path] = field(default_factory=list)
    hamming_threshold: int = 8
    cosine_threshold: float = 0.2
    ssim_threshold: float = 0.9
    model_name: str = "clip-vit"


__all__ = ["PipelineSettings"]
