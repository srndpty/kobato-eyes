"""Duplicate indexing pipeline combining perceptual hashes and embeddings."""

from __future__ import annotations

import logging
import sqlite3
import time
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

logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        dim = int(getattr(self.embedder, "embedding_dim"))
        if not self.index.is_initialized:
            self.index.build(dim, self.initial_capacity)
        else:
            cur = self.index.dim
            if cur is not None and cur != dim:
                raise RuntimeError(f"HNSW dim mismatch: index={cur}, embedder={dim}")

    def _ensure_index(self, dim: int, additional: int) -> None:
        slack = 256
        if not self.index.is_initialized:
            capacity = max(self.initial_capacity, additional + slack)
            self.index.build(dim, capacity)
        else:
            need = self.index.current_count + additional + slack
            self.index.ensure_capacity(need)

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
        if embeddings.ndim != 2:
            raise RuntimeError("Embedder must return a 2D array")
        if embeddings.shape[0] != len(entries):
            raise RuntimeError("Embedder returned mismatched batch size")

        self._ensure_index(embeddings.shape[1], len(entries))

        file_ids: list[int] = []
        vectors: list[np.ndarray] = []
        labels: list[int] = []
        indexed_time = time.time()

        for entry, vector in zip(entries, embeddings):
            file_id = upsert_file(
                self.conn,
                path=str(entry["path"]),
                size=entry["size"],
                mtime=entry["mtime"],
                sha256=entry["sha"],
                width=entry["image"].width,
                height=entry["image"].height,
                indexed_at=indexed_time,
            )
            upsert_signatures(
                self.conn,
                file_id=file_id,
                phash_u64=entry["phash"],
                dhash_u64=entry["dhash"],
            )
            # ループ内（各画像ごと）
            normalized = np.asarray(vector, dtype=np.float32)
            if normalized.ndim != 1:
                normalized = normalized.reshape(-1)
            norm = float(np.linalg.norm(normalized))
            if norm > 0:
                normalized = normalized / norm
            if not normalized.flags.c_contiguous:
                normalized = np.ascontiguousarray(normalized, dtype=np.float32)

            # ① DB へ保存（正規化済みを保存）
            upsert_embedding(
                self.conn,
                file_id=file_id,
                model=self.model_name,
                dim=normalized.shape[0],
                vector=normalized.tobytes(),
            )

            # ② HNSW へ追加するためのバッファにも同じものを積む
            vectors.append(normalized)
            labels.append(file_id)
            file_ids.append(file_id)

        if vectors:
            mat = np.vstack(vectors).astype(np.float32, copy=False)
            # 正規化（念のため）
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat /= norms
            # ラベル重複を除去（DBの file_id はユニークだが、念のため）
            uniq = {}
            for v, lid in zip(mat, labels):
                if lid not in uniq:
                    uniq[lid] = v
            if uniq:
                lids = list(uniq.keys())
                vecs = np.vstack([uniq[lid] for lid in lids])
                self.index.add(vecs, lids, num_threads=1)

        # ★追加：この関数の最後でコミットしておく
        self.conn.commit()

        return file_ids

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(target)


def load_hnsw_index(
    path: str | Path,
    *,
    dim: int,
    space: str = "cosine",
    initial_capacity: int = 2048,
) -> HNSWIndex:
    target = Path(path)
    if target.exists():
        index = HNSWIndex.load(target)
        index.ensure_capacity(index.current_count + initial_capacity)
    else:
        index = HNSWIndex(space=space)
        index.build(dim, max(initial_capacity, dim))
    return index


def add_embeddings_to_hnsw(
    index: HNSWIndex,
    additions: Sequence[tuple[int, np.ndarray]],
    *,
    dim: int,
    initial_capacity: int = 2048,
    num_threads: int = 1,
) -> int:
    """Add embedding vectors to the provided HNSW index."""
    if not additions:
        return 0
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for file_id, vector in additions:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("Embeddings must be one-dimensional vectors")
        vectors.append(array)
        labels.append(int(file_id))
    matrix = np.vstack(vectors)
    additional = len(labels)
    if not index.is_initialized:
        capacity = max(initial_capacity, index.current_count + additional + 256)
        index.build(dim, capacity)
    else:
        current_dim = index.dim
        if current_dim is not None and current_dim != dim:
            raise ValueError("Embedding dimension does not match the index configuration")
        index.ensure_capacity(index.current_count + additional + 256)
    index.add(matrix, labels, num_threads=num_threads)
    return additional


def save_hnsw_index(index: HNSWIndex, path: str | Path) -> None:
    """Persist the index to the filesystem, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    index.save(target)


__all__ = [
    "DuplicateIndexer",
    "EmbedderProtocol",
    "load_hnsw_index",
    "add_embeddings_to_hnsw",
    "save_hnsw_index",
]
