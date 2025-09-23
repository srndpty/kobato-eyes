"""Benchmark throughput of tagging, embedding, and search operations."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from db.connection import get_conn
from db.schema import apply_schema
from dup.indexer import DuplicateIndexer, EmbedderProtocol
from index.hnsw import HNSWIndex
from sig.embedder import OpenClipEmbedder
from tagger.base import ITagger
from utils.image_io import safe_load_image


def _timed(fn: Callable[[], None], repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def _print_stats(label: str, timings: Sequence[float], units: str = "s") -> None:
    mean = statistics.mean(timings)
    stdev = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    best = min(timings)
    worst = max(timings)
    print(
        f"{label:<16} mean={mean:.4f}{units} stdev={stdev:.4f}{units} fastest={best:.4f}{units} slowest={worst:.4f}{units}"
    )


def benchmark_tagger(tagger: ITagger, images: list[Image.Image], repeats: int) -> None:
    timings = _timed(lambda: tagger.infer_batch(images), repeats)
    _print_stats("Tagger", timings)


def benchmark_embedder(embedder: EmbedderProtocol, images: list[Image.Image], repeats: int) -> None:
    timings = _timed(lambda: embedder.embed_images(images), repeats)
    _print_stats("Embedder", timings)


def benchmark_search(index: HNSWIndex, vectors: np.ndarray, repeats: int, k: int) -> None:
    timings = _timed(lambda: index.knn_query(vectors, k=k), repeats)
    _print_stats("Search", timings)


def prepare_index(
    conn_path: Path, embedder: EmbedderProtocol, images: list[Image.Image], model: str
) -> tuple[HNSWIndex, list[int]]:
    conn = get_conn(conn_path)
    apply_schema(conn)
    index = HNSWIndex(space="cosine")
    index.build(dim=embedder.embed_images([images[0]]).shape[1], max_elements=len(images) * 2)
    indexer = DuplicateIndexer(conn, embedder, index, model_name=model)
    # Create fake paths to satisfy indexer; images are in-memory, so dump temporarily
    tmp_dir = conn_path.parent
    paths: list[str] = []
    for idx, image in enumerate(images):
        path = tmp_dir / f"bench_{idx}.png"
        image.save(path, format="PNG")
        paths.append(str(path))
    file_ids = indexer.index_paths(paths)
    conn.close()
    return index, file_ids


def load_images(paths: Iterable[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        image = safe_load_image(path)
        if image is not None:
            images.append(image)
    if not images:
        raise SystemExit("No valid images found for benchmarking")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Image files to benchmark")
    parser.add_argument("--repeats", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--model", default="clip-vit", help="Model name for embedder")
    parser.add_argument("--db", type=Path, default=Path("bench.db"), help="Temporary database path")
    args = parser.parse_args()

    images = load_images(args.images)

    print(f"Loaded {len(images)} image(s)")

    tagger: ITagger = OpenClipEmbedder(args.model, "laion2b_s32b_b79k")  # type: ignore[assignment]
    embedder: EmbedderProtocol = tagger  # OpenClip embedder doubles as tagger if desired

    benchmark_tagger(tagger, images, args.repeats)
    benchmark_embedder(embedder, images, args.repeats)

    index, _ = prepare_index(args.db, embedder, images, model=args.model)
    vectors = embedder.embed_images(images)
    benchmark_search(index, vectors, args.repeats, k=min(10, len(images)))


if __name__ == "__main__":
    main()
