from __future__ import annotations

from pathlib import Path

from dup.scanner import DuplicateFile, DuplicateScanConfig, DuplicateScanner


def make_file(
    file_id: int,
    *,
    path: str,
    size: int,
    width: int,
    height: int,
    phash: int,
    embedding: tuple[float, ...] | None = None,
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
        ),
        make_file(
            2,
            path="b.png",
            size=2_000,
            width=640,
            height=480,
            phash=base_hash ^ 0x1,
        ),
        make_file(
            3,
            path="c.jpg",
            size=1_500,
            width=800,
            height=600,
            phash=base_hash ^ 0x2,
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
    files = [
        make_file(1, path="small.jpg", size=100, width=100, height=100, phash=base_hash),
        make_file(
            2,
            path="large.jpg",
            size=1_000,
            width=100,
            height=100,
            phash=base_hash,
            embedding=(0.9, 0.1),
        ),
        make_file(
            3,
            path="cosine_a.jpg",
            size=800,
            width=200,
            height=200,
            phash=base_hash ^ 0x1,
            embedding=(0.89, 0.11),
        ),
        make_file(
            4,
            path="cosine_b.jpg",
            size=820,
            width=200,
            height=200,
            phash=base_hash ^ 0x2,
            embedding=(0.88, 0.12),
        ),
        make_file(
            5,
            path="cosine_bad.jpg",
            size=830,
            width=200,
            height=200,
            phash=base_hash ^ 0x3,
            embedding=(-0.5, 0.3),
        ),
    ]
    scanner = DuplicateScanner(
        DuplicateScanConfig(hamming_threshold=4, size_ratio=0.5, cosine_threshold=0.9)
    )
    clusters = scanner.build_clusters(files)
    assert len(clusters) == 1
    ids = {entry.file.file_id for entry in clusters[0].files}
    assert ids == {2, 3, 4}
