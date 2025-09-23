"""Tests for the duplicate indexer pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pytest
from PIL import Image

from db.connection import get_conn
from db.schema import apply_schema
from dup.indexer import DuplicateIndexer
from index.hnsw import HNSWIndex
from utils.image_io import safe_load_image


class DummyEmbedder:
    def __init__(self, dim: int = 16) -> None:
        self.embedding_dim = dim

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        vectors = []
        for image in images:
            data = np.asarray(image).astype(np.float32).flatten()
            if data.size < self.embedding_dim:
                data = np.pad(data, (0, self.embedding_dim - data.size))
            else:
                data = data[: self.embedding_dim]
            norm = np.linalg.norm(data)
            if norm > 0:
                data /= norm
            vectors.append(data.astype(np.float32))
        return np.vstack(vectors)


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (32, 32), color=color)
    image.save(path, format="PNG")


@pytest.fixture()
def memory_conn() -> Iterable[sqlite3.Connection]:
    conn = get_conn(":memory:")
    apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_duplicate_indexer_stores_embeddings(memory_conn: sqlite3.Connection, tmp_path: Path) -> None:
    paths = []
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for index, colour in enumerate(colours):
        path = tmp_path / f"sample_{index}.png"
        _make_image(path, colour)
        paths.append(path)

    embedder = DummyEmbedder(dim=32)
    hnsw = HNSWIndex(space="cosine")
    indexer = DuplicateIndexer(memory_conn, embedder, hnsw, model_name="dummy", initial_capacity=4)

    file_ids = indexer.index_paths(paths)
    assert len(file_ids) == len(paths)

    embeddings_count = memory_conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert embeddings_count == len(paths)

    signatures_count = memory_conn.execute("SELECT COUNT(*) FROM signatures").fetchone()[0]
    assert signatures_count == len(paths)

    hnsw.set_ef(32)
    images = [safe_load_image(path) for path in paths]
    assert all(image is not None for image in images)
    vectors = embedder.embed_images(images)  # type: ignore[arg-type]
    labels, _ = hnsw.knn_query(vectors, k=1)
    assert labels[:, 0].tolist() == file_ids
