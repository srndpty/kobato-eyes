"""Benchmark throughput of tagging, embedding, and search operations."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image

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


if __name__ == "__main__":
    main()
