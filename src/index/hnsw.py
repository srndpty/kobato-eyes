"""Stub HNSW index implementation for unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence


@dataclass
class HnswIndex:
    """Minimal stub used in unit tests to satisfy imports."""

    path: Path
    dim: int

    def add_items(self, embeddings: Sequence[Sequence[float]]) -> None:  # pragma: no cover - stub
        """Pretend to add embedding vectors."""

    def save(self) -> None:  # pragma: no cover - stub
        """Pretend to persist the index to disk."""


hnswlib = SimpleNamespace(Index=HnswIndex)


__all__ = ["HnswIndex", "hnswlib"]
