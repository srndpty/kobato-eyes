"""Duplicate indexing pipeline combining perceptual hashes and embeddings."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from db.repository import upsert_embedding, upsert_file, upsert_signatures
from index.hnsw import HNSWIndex
from sig.phash import dhash, phash
from utils.hash import compute_sha256
from utils.image_io import safe_load_image


class EmbedderProtocol(Protocol):
    """Protocol describing the required embedder interface."""

    @property
    def embedding_dim(self) -> int:  # pragma: no cover - protocol declaration
        ...

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class DuplicateIndexer:
    """Orchestrate duplicate indexing by persisting hashes and embeddings."""

    conn: sqlite3.Connection
    embedder: EmbedderProtocol
    index: HNSWIndex
    model_name: str
    initial_capacity: int = 2048

    def _ensure_index(self, dim: int, additional: int) -> None:
        if not self.index.is_initialized:
            capacity = max(self.initial_capacity, additional)
            self.index.build(dim, capacity)
        else:
            needed = self.index.current_count + additional
            self.index.ensure_capacity(needed)

    def index_paths(self, paths: Sequence[str | Path]) -> list[int]:
        """Compute hashes/embeddings for ``paths`` and persist them."""
        entries: list[dict] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                continue
            image = safe_load_image(path)
            if image is None:
                continue
            stat = path.stat()
            entry = {
                "path": path,
                "image": image,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha": compute_sha256(path),
                "phash": phash(image),
                "dhash": dhash(image),
            }
            entries.append(entry)

        if not entries:
            return []

        embeddings = self.embedder.embed_images([entry["image"] for entry in entries])
        if embeddings.shape[0] != len(entries):
            raise RuntimeError("Embedder returned mismatched batch size")

        self._ensure_index(embeddings.shape[1], len(entries))

        file_ids: list[int] = []
        vectors: list[np.ndarray] = []
        labels: list[int] = []

        for entry, vector in zip(entries, embeddings):
            file_id = upsert_file(
                self.conn,
                path=str(entry["path"]),
                size=entry["size"],
                mtime=entry["mtime"],
                sha256=entry["sha"],
            )
            upsert_signatures(
                self.conn,
                file_id=file_id,
                phash_u64=entry["phash"],
                dhash_u64=entry["dhash"],
            )
            normalized = vector.astype(np.float32)
            # Safeguard: ensure unit length before persistence/indexing.
            norm = float(np.linalg.norm(normalized))
            if norm > 0:
                normalized /= norm
            upsert_embedding(
                self.conn,
                file_id=file_id,
                model=self.model_name,
                dim=normalized.shape[0],
                vector=normalized.tobytes(),
            )
            file_ids.append(file_id)
            vectors.append(normalized)
            labels.append(file_id)

        if vectors:
            self.index.add(np.vstack(vectors), labels)

        return file_ids


__all__ = ["DuplicateIndexer", "EmbedderProtocol"]
