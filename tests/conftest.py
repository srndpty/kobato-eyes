"""Shared pytest fixtures."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class _StubHNSWIndex:
    def __init__(self, space: str, dim: int) -> None:
        self.space = space
        self.dim = dim
        self._max_elements = 0
        self._vectors: dict[int, np.ndarray] = {}
        self._ef = 0

    def init_index(self, max_elements: int, ef_construction: int, M: int) -> None:  # noqa: N803
        self._max_elements = max_elements

    def resize_index(self, total_capacity: int) -> None:
        self._max_elements = total_capacity

    def add_items(self, data: np.ndarray, labels: np.ndarray, num_threads: int = 1) -> None:
        for vector, label in zip(data, labels):
            self._vectors[int(label)] = np.array(vector, dtype=np.float32)

    def get_current_count(self) -> int:
        return len(self._vectors)

    def set_ef(self, ef: int) -> None:
        self._ef = ef

    def knn_query(self, data: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(data, dtype=np.float32)
        labels_out = []
        distances_out = []
        stored = list(self._vectors.items())
        for query in queries:
            scores: list[tuple[float, int]] = []
            for label, vector in stored:
                if self.space == "cosine":
                    score = 1.0 - float(np.dot(query, vector))
                else:
                    score = float(np.linalg.norm(query - vector))
                scores.append((score, label))
            scores.sort(key=lambda item: item[0])
            selected = scores[:k]
            if len(selected) < k:
                selected.extend([(float("inf"), -1)] * (k - len(selected)))
            distances_out.append([score for score, _ in selected])
            labels_out.append([label for _, label in selected])
        return np.asarray(labels_out, dtype=np.int64), np.asarray(distances_out, dtype=np.float32)

    def save_index(self, path: str) -> None:
        raise NotImplementedError

    def load_index(self, path: str) -> None:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def stub_hnsw(monkeypatch: pytest.MonkeyPatch) -> None:
    from index import hnsw as hnsw_module

    stub_module = SimpleNamespace(Index=_StubHNSWIndex)
    monkeypatch.setattr(hnsw_module, "hnswlib", stub_module)
