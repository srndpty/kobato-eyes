"""Tests for duplicate candidate discovery."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from db.connection import get_conn
from db.repository import upsert_embedding, upsert_file, upsert_signatures
from db.schema import apply_schema
from dup.candidates import CandidateFinder
from index.hnsw import HNSWIndex


@pytest.fixture()
def memory_conn() -> sqlite3.Connection:
    conn = get_conn(":memory:")
    apply_schema(conn)
    return conn


def _insert_file(
    conn: sqlite3.Connection,
    *,
    path: Path,
    size: int,
    mtime: float,
    sha: str,
    phash_value: int,
    dhash_value: int,
    embedding: np.ndarray,
    model: str,
) -> int:
    file_id = upsert_file(conn, path=str(path), size=size, mtime=mtime, sha256=sha)
    upsert_signatures(conn, file_id=file_id, phash_u64=phash_value, dhash_u64=dhash_value)
    upsert_embedding(
        conn,
        file_id=file_id,
        model=model,
        dim=embedding.shape[0],
        vector=embedding.astype(np.float32).tobytes(),
    )
    return file_id


def test_candidate_finder_combines_stages(memory_conn: sqlite3.Connection, tmp_path: Path) -> None:
    model = "dummy"
    vectors = {
        "a": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        "c": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }

    ids: dict[str, int] = {}
    ids["a"] = _insert_file(
        memory_conn,
        path=tmp_path / "a.png",
        size=10,
        mtime=0.0,
        sha="sha_a",
        phash_value=0b11110000,
        dhash_value=0,
        embedding=vectors["a"],
        model=model,
    )
    ids["b"] = _insert_file(
        memory_conn,
        path=tmp_path / "b.png",
        size=10,
        mtime=0.0,
        sha="sha_b",
        phash_value=0b11110001,
        dhash_value=0,
        embedding=vectors["b"],
        model=model,
    )
    ids["c"] = _insert_file(
        memory_conn,
        path=tmp_path / "c.png",
        size=10,
        mtime=0.0,
        sha="sha_c",
        phash_value=0b00001111,
        dhash_value=0,
        embedding=vectors["c"],
        model=model,
    )

    index = HNSWIndex(space="cosine")
    index.build(dim=4, max_elements=10)
    index.add(np.vstack(list(vectors.values())), list(ids.values()))

    finder = CandidateFinder(memory_conn, index, model_name=model)
    candidates = finder.find_for_file(ids["a"], hamming_threshold=2, top_k=3, ef=16)

    assert any(candidate.file_id == ids["b"] for candidate in candidates)
    phash_candidate = next(candidate for candidate in candidates if candidate.file_id == ids["b"])
    assert phash_candidate.phash_distance == 1
    assert phash_candidate.cosine_distance is not None

    # Ensure far vector might appear via knn but with higher cosine distance
    cosine_only = next(candidate for candidate in candidates if candidate.file_id == ids["c"])
    assert cosine_only.phash_distance is None
    assert cosine_only.cosine_distance is not None
