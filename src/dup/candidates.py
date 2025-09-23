"""Candidate generation for duplicate detection."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from index.hnsw import HNSWIndex
from sig.phash import hamming64


@dataclass
class Candidate:
    """Candidate duplicate with stage metrics."""

    file_id: int
    phash_distance: Optional[int] = None
    cosine_distance: Optional[float] = None

    def update_phash(self, distance: int) -> None:
        self.phash_distance = distance

    def update_cosine(self, distance: float) -> None:
        self.cosine_distance = distance


class CandidateFinder:
    """Find duplicate candidates using perceptual hashes and embedding index."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        index: HNSWIndex,
        *,
        model_name: str,
    ) -> None:
        self._conn = conn
        self._index = index
        self._model_name = model_name

    def find_for_file(
        self,
        file_id: int,
        *,
        hamming_threshold: int = 8,
        top_k: int = 20,
        ef: int | None = None,
    ) -> list[Candidate]:
        candidates: Dict[int, Candidate] = {}
        self._stage_phash(file_id, hamming_threshold, candidates)
        self._stage_hnsw(file_id, top_k, ef, candidates)
        candidates.pop(file_id, None)
        return sorted(
            candidates.values(),
            key=lambda item: (
                item.phash_distance if item.phash_distance is not None else 1_000,
                item.cosine_distance if item.cosine_distance is not None else 1_000.0,
                item.file_id,
            ),
        )

    def _stage_phash(
        self,
        file_id: int,
        threshold: int,
        candidates: Dict[int, Candidate],
    ) -> None:
        query = "SELECT phash_u64 FROM signatures WHERE file_id = ?"
        row = self._conn.execute(query, (file_id,)).fetchone()
        if row is None:
            return
        source_phash = int(row["phash_u64"])
        for other in self._conn.execute("SELECT file_id, phash_u64 FROM signatures"):
            other_id = int(other["file_id"])
            if other_id == file_id:
                continue
            distance = hamming64(source_phash, int(other["phash_u64"]))
            if distance <= threshold:
                candidate = candidates.setdefault(other_id, Candidate(file_id=other_id))
                candidate.update_phash(distance)

    def _stage_hnsw(
        self,
        file_id: int,
        top_k: int,
        ef: int | None,
        candidates: Dict[int, Candidate],
    ) -> None:
        query = "SELECT vector, dim FROM embeddings WHERE file_id = ? AND model = ?"
        row = self._conn.execute(query, (file_id, self._model_name)).fetchone()
        if row is None:
            return
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        dim = int(row["dim"])
        if vector.size != dim:
            vector = vector[:dim]
        if ef is not None:
            self._index.set_ef(ef)
        labels, distances = self._index.knn_query(vector.reshape(1, -1), k=top_k)
        for label, distance in zip(labels[0], distances[0]):
            other_id = int(label)
            if other_id < 0 or other_id == file_id:
                continue
            candidate = candidates.setdefault(other_id, Candidate(file_id=other_id))
            candidate.update_cosine(float(distance))


__all__ = ["Candidate", "CandidateFinder"]
