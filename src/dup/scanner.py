"""Duplicate scanning pipeline using LSH buckets and DSU clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from sig.phash import hamming64


_EXTENSION_PRIORITY = {
    "png": 4,
    "apng": 4,
    "webp": 3,
    "tiff": 2,
    "tif": 2,
    "bmp": 1,
    "gif": 1,
    "jpeg": 0,
    "jpg": 0,
    "jpe": 0,
    "jfif": 0,
}


@dataclass(frozen=True)
class DuplicateFile:
    """Metadata required for duplicate detection of a single file."""

    file_id: int
    path: Path
    size: int | None
    width: int | None
    height: int | None
    phash: int
    embedding: np.ndarray | None

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> DuplicateFile:
        """Create an instance from a database row."""

        phash_value = row.get("phash_u64")
        if phash_value is None:
            raise ValueError("Row is missing perceptual hash information")
        raw_path = Path(str(row.get("path")))
        size = row.get("size")
        width = row.get("width")
        height = row.get("height")
        embedding: np.ndarray | None = None
        raw_embedding = row.get("embedding_vector")
        dim_value = row.get("embedding_dim")
        if raw_embedding is not None and dim_value is not None:
            dim = int(dim_value)
            buffer = raw_embedding
            if isinstance(buffer, memoryview):
                buffer = buffer.tobytes()
            array = np.frombuffer(buffer, dtype=np.float32)
            if dim > 0:
                array = array[:dim]
            embedding = array.astype(np.float32, copy=True) if array.size else None
        return cls(
            file_id=int(row.get("file_id")),
            path=raw_path,
            size=int(size) if isinstance(size, (int, float)) else None,
            width=int(width) if isinstance(width, (int, float)) else None,
            height=int(height) if isinstance(height, (int, float)) else None,
            phash=int(phash_value),
            embedding=embedding,
        )

    @property
    def resolution(self) -> int:
        width = self.width or 0
        height = self.height or 0
        return width * height

    @property
    def extension_priority(self) -> int:
        suffix = self.path.suffix.lower().lstrip(".")
        return _EXTENSION_PRIORITY.get(suffix, 0)


@dataclass(frozen=True)
class DuplicateClusterEntry:
    """Single file inside a duplicate cluster."""

    file: DuplicateFile
    best_hamming: int | None
    best_cosine: float | None


@dataclass(frozen=True)
class DuplicateCluster:
    """Duplicate cluster along with the keeper identifier."""

    files: list[DuplicateClusterEntry]
    keeper_id: int


@dataclass(frozen=True)
class DuplicateScanConfig:
    """Configuration values controlling duplicate detection thresholds."""

    hamming_threshold: int = 8
    size_ratio: float | None = None
    cosine_threshold: float | None = None
    band_bits: int = 16
    band_count: int = 4

    def __post_init__(self) -> None:
        if self.band_bits <= 0:
            raise ValueError("band_bits must be positive")
        if self.band_count <= 0:
            raise ValueError("band_count must be positive")
        if self.hamming_threshold < 0 or self.hamming_threshold > 64:
            raise ValueError("hamming_threshold must be in [0, 64]")


@dataclass
class DuplicateEdge:
    file_id_a: int
    file_id_b: int
    hamming: int | None
    cosine: float | None


class DisjointSet:
    """Disjoint-set union data structure for clustering."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def find(self, item: int) -> int:
        parent = self._parent.setdefault(item, item)
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self._rank.get(root_a, 0)
        rank_b = self._rank.get(root_b, 0)
        if rank_a < rank_b:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if rank_a == rank_b:
            self._rank[root_a] = rank_a + 1


class DuplicateScanner:
    """Generate duplicate clusters using LSH banding and DSU clustering."""

    def __init__(self, config: DuplicateScanConfig) -> None:
        self._config = config
        self._band_mask = (1 << config.band_bits) - 1

    def build_clusters(self, files: Iterable[DuplicateFile]) -> list[DuplicateCluster]:
        """Return clusters of duplicate files."""

        candidates = [file for file in files if file.phash is not None]
        if not candidates:
            return []

        buckets: dict[tuple[int, int], list[int]] = {}
        for index, file in enumerate(candidates):
            for band_index in range(self._config.band_count):
                shift = band_index * self._config.band_bits
                bucket_key = (band_index, (file.phash >> shift) & self._band_mask)
                buckets.setdefault(bucket_key, []).append(index)

        edges: dict[tuple[int, int], DuplicateEdge] = {}
        for indices in buckets.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices) - 1):
                idx_a = indices[i]
                file_a = candidates[idx_a]
                for j in range(i + 1, len(indices)):
                    idx_b = indices[j]
                    if idx_a == idx_b:
                        continue
                    key = tuple(sorted((file_a.file_id, candidates[idx_b].file_id)))
                    if key in edges:
                        continue
                    file_b = candidates[idx_b]
                    if not self._passes_size_ratio(file_a, file_b):
                        continue
                    hamming = hamming64(file_a.phash, file_b.phash)
                    if hamming > self._config.hamming_threshold:
                        continue
                    cosine = self._compute_cosine_distance(file_a.embedding, file_b.embedding)
                    if (
                        self._config.cosine_threshold is not None
                        and cosine is not None
                        and cosine > self._config.cosine_threshold
                    ):
                        continue
                    edges[key] = DuplicateEdge(
                        file_id_a=file_a.file_id,
                        file_id_b=file_b.file_id,
                        hamming=hamming,
                        cosine=cosine,
                    )

        if not edges:
            return []

        dsu = DisjointSet()
        files_by_id = {file.file_id: file for file in candidates}
        best_hamming: dict[int, int] = {}
        best_cosine: dict[int, float] = {}
        for edge in edges.values():
            dsu.union(edge.file_id_a, edge.file_id_b)
            for file_id, distance in ((edge.file_id_a, edge.hamming), (edge.file_id_b, edge.hamming)):
                if distance is not None:
                    current = best_hamming.get(file_id)
                    if current is None or distance < current:
                        best_hamming[file_id] = distance
            for file_id, distance in ((edge.file_id_a, edge.cosine), (edge.file_id_b, edge.cosine)):
                if distance is not None:
                    current = best_cosine.get(file_id)
                    if current is None or distance < current:
                        best_cosine[file_id] = distance

        groups: dict[int, list[int]] = {}
        for file_id in {fid for edge in edges.values() for fid in (edge.file_id_a, edge.file_id_b)}:
            root = dsu.find(file_id)
            groups.setdefault(root, []).append(file_id)

        clusters: list[DuplicateCluster] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            entries: list[DuplicateClusterEntry] = []
            for file_id in sorted(members):
                file = files_by_id.get(file_id)
                if file is None:
                    continue
                entries.append(
                    DuplicateClusterEntry(
                        file=file,
                        best_hamming=best_hamming.get(file_id),
                        best_cosine=best_cosine.get(file_id),
                    )
                )
            if len(entries) < 2:
                continue
            keeper_id = self._choose_keeper(entries)
            entries.sort(
                key=lambda entry: (
                    0 if entry.file.file_id == keeper_id else 1,
                    -(entry.file.size or 0),
                    -entry.file.resolution,
                    -entry.file.extension_priority,
                    entry.file.path.name.lower(),
                    entry.file.file_id,
                )
            )
            clusters.append(DuplicateCluster(files=entries, keeper_id=keeper_id))

        clusters.sort(
            key=lambda cluster: (
                -(max(entry.file.size or 0 for entry in cluster.files)),
                cluster.files[0].file.path.as_posix().lower(),
            )
        )
        return clusters

    def _passes_size_ratio(self, left: DuplicateFile, right: DuplicateFile) -> bool:
        ratio = self._config.size_ratio
        if ratio is None or ratio <= 0:
            return True
        left_size = left.size or 0
        right_size = right.size or 0
        if left_size <= 0 or right_size <= 0:
            return True
        smaller = min(left_size, right_size)
        larger = max(left_size, right_size)
        if larger == 0:
            return True
        return (smaller / larger) >= ratio

    @staticmethod
    def _compute_cosine_distance(
        left: np.ndarray | None, right: np.ndarray | None
    ) -> float | None:
        if left is None or right is None:
            return None
        dim = min(left.shape[0], right.shape[0])
        if dim == 0:
            return None
        vec_a = left[:dim]
        vec_b = right[:dim]
        dot = float(np.dot(vec_a, vec_b))
        distance = 1.0 - dot
        if distance < 0.0:
            distance = 0.0
        if distance > 2.0:
            distance = 2.0
        return distance

    @staticmethod
    def _choose_keeper(entries: Sequence[DuplicateClusterEntry]) -> int:
        def key(entry: DuplicateClusterEntry) -> tuple:
            file = entry.file
            return (
                -(file.size or 0),
                -file.resolution,
                -file.extension_priority,
                file.path.suffix.lower(),
                file.path.name.lower(),
                file.file_id,
            )

        return min(entries, key=key).file.file_id


__all__ = [
    "DuplicateFile",
    "DuplicateCluster",
    "DuplicateClusterEntry",
    "DuplicateScanConfig",
    "DuplicateScanner",
]
