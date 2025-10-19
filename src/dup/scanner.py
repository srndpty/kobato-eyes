"""Duplicate scanning pipeline using LSH buckets and DSU clustering."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sig.phash import hamming64

logger = logging.getLogger(__name__)

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


def _row_get(row, key, default=None):
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        if hasattr(row, "keys") and key in row.keys():
            return row[key]  # sqlite3.Row
        return getattr(row, key, default)
    except Exception:
        return default


def _parse_phash_any(raw) -> int | None:
    if raw is None:
        return None

    # memoryview / bytes / bytearray (BLOB)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            v = int.from_bytes(bytes(raw), "big", signed=False)
            return v & ((1 << 64) - 1)
        except Exception:
            return None

    # 文字列 (10進/0x16進/16進っぽい)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            # base=0 なら "0x…" も 10進も自動判定
            v = int(s, 0)
        except Exception:
            try:
                v = int(s, 16)  # 最後の手段として16進解釈
            except Exception:
                return None
        return v & ((1 << 64) - 1)

    # それ以外（np.int64 / Decimal などもここで拾う）
    try:
        v = int(raw)
    except Exception:
        return None
    return v & ((1 << 64) - 1)


PHASH_KEYS = ("phash_u64", "phash", "phash64", "phash_hex", "phash_bytes", "signature", "sig")


@dataclass(frozen=True)
class DuplicateFile:
    """Metadata required for duplicate detection of a single file."""

    file_id: int
    path: Path
    size: int | None
    width: int | None
    height: int | None
    phash: int
    embedding: tuple[float, ...] | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> DuplicateFile:
        raw_ph = None
        for k in PHASH_KEYS:
            raw_ph = _row_get(row, k, None)
            if raw_ph is not None:
                break
        ph = _parse_phash_any(raw_ph)
        if ph is None:
            raise ValueError("Row is missing perceptual hash information")

        return cls(
            file_id=int(_row_get(row, "file_id", _row_get(row, "id", -1))),
            path=Path(str(_row_get(row, "path", _row_get(row, "file_path", "")))),
            size=(lambda v: int(v) if isinstance(v, (int, float)) else None)(_row_get(row, "size")),
            width=(lambda v: int(v) if isinstance(v, (int, float)) else None)(_row_get(row, "width")),
            height=(lambda v: int(v) if isinstance(v, (int, float)) else None)(_row_get(row, "height")),
            phash=ph,
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
    band_bits: int = 16
    band_count: int = 4
    cosine_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.band_bits <= 0:
            raise ValueError("band_bits must be positive")
        if self.band_count <= 0:
            raise ValueError("band_count must be positive")
        if self.hamming_threshold < 0 or self.hamming_threshold > 64:
            raise ValueError("hamming_threshold must be in [0, 64]")
        if self.cosine_threshold is not None:
            if not (-1.0 <= self.cosine_threshold <= 1.0):
                raise ValueError("cosine_threshold must be between -1.0 and 1.0")


@dataclass
class DuplicateEdge:
    file_id_a: int
    file_id_b: int
    hamming: int | None


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
        assert config.band_bits * config.band_count <= 64, "band config too large"
        self._band_mask = (1 << config.band_bits) - 1

    def build_clusters(self, files: Iterable[DuplicateFile]) -> list[DuplicateCluster]:
        candidates = [f for f in files if f.phash is not None]
        logger.info(
            "dup: candidates=%d band_bits=%d band_count=%d ham_th=%d size_ratio=%s cosine_th=%s",
            len(candidates),
            self._config.band_bits,
            self._config.band_count,
            self._config.hamming_threshold,
            self._config.size_ratio,
            self._config.cosine_threshold,
        )

        if not candidates:
            return []

        # --- バケット生成 ---
        buckets: dict[tuple[int, int], list[int]] = {}
        for idx, f in enumerate(candidates):
            ph = f.phash & ((1 << 64) - 1)  # 念のため
            for b in range(self._config.band_count):
                shift = b * self._config.band_bits
                key = (b, (ph >> shift) & self._band_mask)
                buckets.setdefault(key, []).append(idx)

        bucket_sizes = [len(v) for v in buckets.values()]
        ge2 = sum(1 for s in bucket_sizes if s >= 2)
        max_bucket = max(bucket_sizes) if bucket_sizes else 0
        logger.info("dup: buckets=%d (>=2:%d) max_bucket=%d", len(buckets), ge2, max_bucket)
        if ge2 == 0:
            logger.warning("dup: no bucket has 2+ items -> edges=0")
            return []

        # --- ペア生成とフィルタの通過数を計測 ---
        edges: dict[tuple[int, int], DuplicateEdge] = {}
        pair_total = pair_after_size = pair_after_ham = pair_after_cos = 0

        for indices in buckets.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices) - 1):
                a = candidates[indices[i]]
                for j in range(i + 1, len(indices)):
                    b = candidates[indices[j]]
                    if a.file_id == b.file_id:
                        continue
                    pair_total += 1

                    if not self._passes_size_ratio(a, b):
                        continue
                    pair_after_size += 1

                    h = hamming64(a.phash, b.phash)
                    if h > self._config.hamming_threshold:
                        continue
                    pair_after_ham += 1

                    if not self._passes_cosine_similarity(a, b):
                        continue
                    pair_after_cos += 1

                    key = (a.file_id, b.file_id) if a.file_id < b.file_id else (b.file_id, a.file_id)
                    if key not in edges:
                        edges[key] = DuplicateEdge(a.file_id, b.file_id, h)

        logger.info(
            "dup: pairs total=%d -> size=%d -> ham=%d -> cosine=%d -> edges=%d",
            pair_total,
            pair_after_size,
            pair_after_ham,
            pair_after_cos,
            len(edges),
        )

        if not edges:
            return []

        dsu = DisjointSet()
        files_by_id = {file.file_id: file for file in candidates}
        best_hamming: dict[int, int] = {}
        for edge in edges.values():
            dsu.union(edge.file_id_a, edge.file_id_b)
            for file_id, distance in ((edge.file_id_a, edge.hamming), (edge.file_id_b, edge.hamming)):
                if distance is not None:
                    current = best_hamming.get(file_id)
                    if current is None or distance < current:
                        best_hamming[file_id] = distance

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

    def _passes_cosine_similarity(self, left: DuplicateFile, right: DuplicateFile) -> bool:
        """Return True when cosine similarity is above the configured threshold."""

        threshold = self._config.cosine_threshold
        if threshold is None:
            return True
        cosine = self._compute_cosine_similarity(left, right)
        if cosine is None:
            return True
        return cosine >= threshold

    @staticmethod
    def _compute_cosine_similarity(left: DuplicateFile, right: DuplicateFile) -> float | None:
        """Compute cosine similarity between two duplicate file embeddings."""

        vec_left = left.embedding
        vec_right = right.embedding
        if vec_left is None or vec_right is None:
            return None
        if len(vec_left) == 0 or len(vec_right) == 0:
            return None
        if len(vec_left) != len(vec_right):
            return None
        dot = sum(a * b for a, b in zip(vec_left, vec_right))
        norm_left = math.sqrt(sum(a * a for a in vec_left))
        norm_right = math.sqrt(sum(b * b for b in vec_right))
        if norm_left == 0.0 or norm_right == 0.0:
            return None
        return dot / (norm_left * norm_right)

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
