"""Tests for the HNSW index wrapper."""

from __future__ import annotations

import numpy as np

from index.hnsw import HNSWIndex


def test_hnsw_self_neighbour() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(5, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = HNSWIndex(space="cosine")
    index.build(dim=8, max_elements=16, ef_construction=200, m=16)
    index.add(vectors, list(range(5)))
    index.set_ef(32)

    labels, distances = index.knn_query(vectors, k=1)
    assert labels.shape == (5, 1)
    assert distances.shape == (5, 1)
    assert labels[:, 0].tolist() == list(range(5))
