"""Wrapper around hnswlib for approximate nearest-neighbour queries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import hnswlib
import numpy as np


class HNSWIndex:
    """Convenience wrapper managing an hnswlib index lifecycle."""

    def __init__(self, *, space: str = "cosine") -> None:
        self._space = space
        self._index: hnswlib.Index | None = None
        self._dim: int | None = None
        self._max_elements: int = 0

    @property
    def is_initialized(self) -> bool:
        return self._index is not None

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def current_count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.get_current_count())

    @property
    def max_elements(self) -> int:
        return self._max_elements

    def build(
        self,
        dim: int,
        max_elements: int,
        *,
        ef_construction: int = 200,
        m: int = 16,
    ) -> None:
        """Initialise a new index capable of holding up to ``max_elements``."""
        self._dim = int(dim)
        self._max_elements = int(max_elements)
        self._index = hnswlib.Index(space=self._space, dim=self._dim)
        self._index.init_index(max_elements=self._max_elements, ef_construction=ef_construction, M=m)

    def ensure_capacity(self, total_capacity: int) -> None:
        """Grow the index if more slots are required."""
        if self._index is None:
            raise RuntimeError("Index is not initialised")
        if total_capacity <= self._max_elements:
            return
        self._index.resize_index(total_capacity)
        self._max_elements = total_capacity

    def add(self, vectors: np.ndarray, ids: Sequence[int], *, num_threads: int = 1) -> None:
        """Add vectors with the given integer identifiers."""
        if self._index is None or self._dim is None:
            raise RuntimeError("Index is not initialised")
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self._dim:
            raise ValueError("Vector dimensionality mismatch")
        labels = np.ascontiguousarray(list(ids), dtype=np.int64)
        if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
            raise ValueError("Each vector must have a matching identifier")
        self._index.add_items(data, labels, num_threads=num_threads)

    def knn_query(
        self,
        vectors: np.ndarray,
        k: int,
        *,
        ef: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the index for the ``k`` nearest neighbours."""
        if self._index is None:
            raise RuntimeError("Index is not initialised")
        if ef is not None:
            self.set_ef(ef)
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        return self._index.knn_query(data, k=k)

    def set_ef(self, ef: int) -> None:
        if self._index is None:
            raise RuntimeError("Index is not initialised")
        self._index.set_ef(int(ef))

    def save(self, path: str | Path) -> None:
        """Serialize the index and metadata to disk."""
        if self._index is None or self._dim is None:
            raise RuntimeError("Index is not initialised")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        self._index.save_index(str(tmp_path))
        os.replace(tmp_path, target)
        meta_path = target.with_suffix(target.suffix + ".meta.json")
        meta = {
            "space": self._space,
            "dim": self._dim,
            "max_elements": self._max_elements,
        }
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with meta_tmp.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        os.replace(meta_tmp, meta_path)

    @classmethod
    def load(cls, path: str | Path) -> "HNSWIndex":
        """Load an index previously persisted with :meth:`save`."""
        target = Path(path)
        meta_path = target.with_suffix(target.suffix + ".meta.json")
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        instance = cls(space=meta["space"])
        instance._dim = int(meta["dim"])
        instance._max_elements = int(meta["max_elements"])
        index = hnswlib.Index(space=instance._space, dim=instance._dim)
        index.load_index(str(target))
        instance._index = index
        return instance


__all__ = ["HNSWIndex"]
