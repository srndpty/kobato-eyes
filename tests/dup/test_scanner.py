from __future__ import annotations

from pathlib import Path

import numpy as np

from dup.scanner import DuplicateFile, DuplicateScanConfig, DuplicateScanner


def make_file(
    file_id: int,
    *,
    path: str,
    size: int,
    width: int,
    height: int,
    phash: int,
    embedding: np.ndarray | None = None,
) -> DuplicateFile:
    return DuplicateFile(
        file_id=file_id,
        path=Path(path),
        size=size,
        width=width,
        height=height,
        phash=phash,
        embedding=embedding,
    )


def test_scanner_clusters_and_keeper_selection() -> None:
    base_hash = 0xFFFF_FFFF_0000_0000
    files = [
        make_file(
            1,
            path="a.jpg",
            size=1_000,
            width=640,
            height=480,
            phash=base_hash,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        make_file(
            2,
            path="b.png",
            size=2_000,
            width=640,
            height=480,
            phash=base_hash ^ 0x1,
            embedding=np.array([0.9, 0.1], dtype=np.float32),
        ),
        make_file(
            3,
            path="c.jpg",
            size=1_500,
            width=800,
            height=600,
            phash=base_hash ^ 0x2,
            embedding=np.array([0.95, 0.05], dtype=np.float32),
        ),
    ]
    scanner = DuplicateScanner(DuplicateScanConfig(hamming_threshold=4))
    clusters = scanner.build_clusters(files)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.keeper_id == 2  # largest file with preferred extension
    file_ids = {entry.file.file_id for entry in cluster.files}
    assert file_ids == {1, 2, 3}
    for entry in cluster.files:
        if entry.file.file_id == 2:
            continue
        assert entry.best_hamming is not None


def test_scanner_honours_ratio_and_cosine_thresholds() -> None:
    base_hash = 0xAAAA_AAAA_AAAA_AAAA
    good_embedding_a = np.array([1.0, 0.0], dtype=np.float32)
    good_embedding_b = np.array([0.8, 0.6], dtype=np.float32)
    bad_embedding = np.array([-1.0, 0.0], dtype=np.float32)
    files = [
        make_file(1, path="small.jpg", size=100, width=100, height=100, phash=base_hash),
        make_file(
            2,
            path="large.jpg",
            size=1_000,
            width=100,
            height=100,
            phash=base_hash,
            embedding=good_embedding_b,
        ),
        make_file(
            3,
            path="cosine_a.jpg",
            size=800,
            width=200,
            height=200,
            phash=base_hash ^ 0x1,
            embedding=good_embedding_a,
        ),
        make_file(
            4,
            path="cosine_b.jpg",
            size=820,
            width=200,
            height=200,
            phash=base_hash ^ 0x2,
            embedding=good_embedding_b,
        ),
        make_file(
            5,
            path="cosine_bad.jpg",
            size=830,
            width=200,
            height=200,
            phash=base_hash ^ 0x3,
            embedding=bad_embedding,
        ),
    ]
    scanner = DuplicateScanner(
        DuplicateScanConfig(hamming_threshold=4, size_ratio=0.5, cosine_threshold=0.5)
    )
    clusters = scanner.build_clusters(files)
    assert len(clusters) == 1
    ids = {entry.file.file_id for entry in clusters[0].files}
    assert ids == {2, 3, 4}
