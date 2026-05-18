"""Benchmark throughput of tagging operations."""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.config.schema import DEFAULT_ALLOW_EXTS
from core.pipeline.loaders import PrefetchLoaderPrepared
from tagger.base import ITagger
from tagger.pixai_onnx import PixaiOnnxTagger
from tagger.wd14_onnx import WD14Tagger, get_available_providers

logger = logging.getLogger(__name__)

_INFER_LOG_RE = re.compile(
    r"infer_batch_prepared batch=(?P<batch>\d+) .*?"
    r"ort=(?P<ort>[0-9.]+)ms post=(?P<post>[0-9.]+)ms total=(?P<total>[0-9.]+)ms"
)
_BENCH_NUMBER_RE = re.compile(r"^(?P<number>\d{3,})-.*\.json$")
_SLUG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class InferLogMetric:
    """Timing parsed from tagger inference logs."""

    batch_size: int
    ort_ms: float
    post_ms: float
    total_ms: float


@dataclass(slots=True)
class BatchMetric:
    """Per-batch benchmark timing."""

    batch_index: int
    image_count: int
    warmup: bool
    wait_batch_seconds: float
    infer_seconds: float
    batch_seconds: float
    ort_ms: float | None = None
    post_ms: float | None = None
    tagger_total_ms: float | None = None


@dataclass(slots=True)
class TaggerBenchmarkResult:
    """Serializable benchmark result."""

    created_at: str
    root: str
    tagger: str
    model: str
    tags_csv: str | None
    selected_images: int
    processed_images: int
    measured_images: int
    failed_images: int
    batch_size: int
    warmup_batches: int
    prefetch_batches: int
    io_workers: int | None
    providers_available: list[str]
    providers_session: list[str]
    total_seconds: float
    measured_seconds: float
    images_per_second: float
    batch_seconds_mean: float | None
    batch_seconds_p50: float | None
    batch_seconds_p95: float | None
    infer_seconds_mean: float | None
    infer_seconds_p50: float | None
    infer_seconds_p95: float | None
    wait_batch_seconds_mean: float | None
    wait_batch_seconds_p50: float | None
    wait_batch_seconds_p95: float | None
    ort_ms_mean: float | None
    ort_ms_p50: float | None
    ort_ms_p95: float | None
    post_ms_mean: float | None
    post_ms_p50: float | None
    post_ms_p95: float | None
    batches: list[BatchMetric]


class _InferenceLogCapture(logging.Handler):
    """Capture tagger timing logs without changing production tagger APIs."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.metrics: list[InferLogMetric] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = _INFER_LOG_RE.search(message)
        if match is None:
            return
        self.metrics.append(
            InferLogMetric(
                batch_size=int(match.group("batch")),
                ort_ms=float(match.group("ort")),
                post_ms=float(match.group("post")),
                total_ms=float(match.group("total")),
            )
        )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    """Return a nearest-rank percentile for a non-empty numeric sequence."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * percentile)
    return ordered[max(0, min(index, len(ordered) - 1))]


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean for a non-empty numeric sequence."""

    if not values:
        return None
    return float(statistics.mean(values))


def _summarize(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    """Return mean, p50, and p95 for ``values``."""

    return (_mean(values), _percentile(values, 0.50), _percentile(values, 0.95))


def _iter_image_paths(root: Path, *, limit: int, extensions: Iterable[str]) -> list[Path]:
    """Return a deterministic list of image paths under ``root``."""

    resolved = root.expanduser()
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    if resolved.is_file():
        if resolved.suffix.lower() not in allowed:
            raise SystemExit(f"Unsupported image extension: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise SystemExit(f"Image root not found: {resolved}")

    paths: list[Path] = []
    for candidate in sorted(resolved.rglob("*"), key=lambda p: str(p).lower()):
        if candidate.is_file() and candidate.suffix.lower() in allowed:
            paths.append(candidate)
            if len(paths) >= limit:
                break
    if not paths:
        raise SystemExit(f"No supported images found under {resolved}")
    return paths


def _create_tagger(kind: str, model_path: Path, tags_csv: Path | None) -> ITagger:
    """Create the requested ONNX tagger."""

    if kind == "pixai":
        return PixaiOnnxTagger(model_path, tags_csv=tags_csv)
    if kind == "wd14":
        return WD14Tagger(model_path, tags_csv=tags_csv)
    raise ValueError(f"Unsupported tagger: {kind}")


def _session_providers(tagger: ITagger) -> list[str]:
    """Return providers used by the tagger session when available."""

    session = getattr(tagger, "_session", None)
    getter = getattr(session, "get_providers", None)
    if not callable(getter):
        return []
    try:
        return [str(provider) for provider in getter()]
    except Exception as exc:
        logger.warning("Failed to read session providers: %s", exc)
        return []


def _benchmark_tagger(args: argparse.Namespace) -> TaggerBenchmarkResult:
    """Run a fixed-size tagger benchmark and return serializable metrics."""

    limit = max(1, int(args.limit))
    batch_size = max(1, int(args.batch_size))
    warmup_batches = max(0, int(args.warmup_batches))
    paths = _iter_image_paths(args.root, limit=limit, extensions=DEFAULT_ALLOW_EXTS)
    tagger = _create_tagger(args.tagger, args.model, args.tags_csv)
    providers_available = get_available_providers()
    providers_session = _session_providers(tagger)

    capture = _InferenceLogCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    old_level = root_logger.level
    if old_level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    loader = PrefetchLoaderPrepared(
        [str(path) for path in paths],
        tagger=tagger,
        batch_size=batch_size,
        prefetch_batches=max(1, int(args.prefetch_batches)),
        io_workers=args.io_workers,
    )
    batches: list[BatchMetric] = []
    processed_images = 0
    failed_images = 0
    total_start = time.perf_counter()

    try:
        iterator = iter(loader)
        batch_index = 0
        while True:
            wait_start = time.perf_counter()
            try:
                batch_paths, batch_np, _sizes = next(iterator)
            except StopIteration:
                break
            wait_seconds = time.perf_counter() - wait_start
            if not batch_paths:
                continue

            batch_index += 1
            before_log_count = len(capture.metrics)
            infer_start = time.perf_counter()
            try:
                results = tagger.infer_batch_prepared(batch_np)
            except Exception as exc:
                failed_images += len(batch_paths)
                logger.exception("Benchmark batch %d failed: %s", batch_index, exc)
                continue
            infer_seconds = time.perf_counter() - infer_start
            processed_images += len(results)

            parsed = capture.metrics[before_log_count:]
            infer_metric = parsed[-1] if parsed else None
            batches.append(
                BatchMetric(
                    batch_index=batch_index,
                    image_count=len(results),
                    warmup=batch_index <= warmup_batches,
                    wait_batch_seconds=wait_seconds,
                    infer_seconds=infer_seconds,
                    batch_seconds=wait_seconds + infer_seconds,
                    ort_ms=infer_metric.ort_ms if infer_metric else None,
                    post_ms=infer_metric.post_ms if infer_metric else None,
                    tagger_total_ms=infer_metric.total_ms if infer_metric else None,
                )
            )
    finally:
        loader.close()
        root_logger.removeHandler(capture)
        root_logger.setLevel(old_level)
        closer = getattr(tagger, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.debug("Tagger close failed after benchmark", exc_info=True)

    total_seconds = time.perf_counter() - total_start
    measured_batches = [batch for batch in batches if not batch.warmup]
    measured_images = sum(batch.image_count for batch in measured_batches)
    measured_seconds = sum(batch.batch_seconds for batch in measured_batches)
    images_per_second = measured_images / measured_seconds if measured_seconds > 0.0 else 0.0

    batch_mean, batch_p50, batch_p95 = _summarize([batch.batch_seconds for batch in measured_batches])
    infer_mean, infer_p50, infer_p95 = _summarize([batch.infer_seconds for batch in measured_batches])
    wait_mean, wait_p50, wait_p95 = _summarize([batch.wait_batch_seconds for batch in measured_batches])
    ort_mean, ort_p50, ort_p95 = _summarize([batch.ort_ms for batch in measured_batches if batch.ort_ms is not None])
    post_mean, post_p50, post_p95 = _summarize(
        [batch.post_ms for batch in measured_batches if batch.post_ms is not None]
    )

    return TaggerBenchmarkResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        root=str(args.root),
        tagger=str(args.tagger),
        model=str(args.model),
        tags_csv=str(args.tags_csv) if args.tags_csv is not None else None,
        selected_images=len(paths),
        processed_images=processed_images,
        measured_images=measured_images,
        failed_images=failed_images + max(0, len(paths) - processed_images - failed_images),
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        prefetch_batches=max(1, int(args.prefetch_batches)),
        io_workers=args.io_workers,
        providers_available=providers_available,
        providers_session=providers_session,
        total_seconds=total_seconds,
        measured_seconds=measured_seconds,
        images_per_second=images_per_second,
        batch_seconds_mean=batch_mean,
        batch_seconds_p50=batch_p50,
        batch_seconds_p95=batch_p95,
        infer_seconds_mean=infer_mean,
        infer_seconds_p50=infer_p50,
        infer_seconds_p95=infer_p95,
        wait_batch_seconds_mean=wait_mean,
        wait_batch_seconds_p50=wait_p50,
        wait_batch_seconds_p95=wait_p95,
        ort_ms_mean=ort_mean,
        ort_ms_p50=ort_p50,
        ort_ms_p95=ort_p95,
        post_ms_mean=post_mean,
        post_ms_p50=post_p50,
        post_ms_p95=post_p95,
        batches=batches,
    )


def _write_json(path: Path, result: TaggerBenchmarkResult) -> None:
    """Write benchmark results as stable, readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(result)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_slug(value: str) -> str:
    """Return a filesystem-friendly benchmark slug."""

    slug = _SLUG_SAFE_RE.sub("-", value.strip()).strip(".-")
    return slug or "run"


def _next_benchmark_number(output_dir: Path) -> int:
    """Return the next numeric prefix for benchmark JSON files."""

    max_number = 0
    if output_dir.exists():
        for candidate in output_dir.glob("*.json"):
            match = _BENCH_NUMBER_RE.match(candidate.name)
            if match is None:
                continue
            max_number = max(max_number, int(match.group("number")))
    return max_number + 1


def _resolve_output_json(args: argparse.Namespace) -> Path:
    """Resolve the benchmark output path from CLI options."""

    if bool(args.auto_number):
        output_dir = Path(args.output_dir)
        number = _next_benchmark_number(output_dir)
        slug = _safe_slug(str(args.run_slug))
        return output_dir / f"{number:03d}-{args.tagger}-{slug}.json"
    if args.output_json is not None:
        return Path(args.output_json)
    return Path("tmp/bench/tagging-baseline.json")


def _fmt_optional(value: float | None, *, digits: int = 3) -> str:
    """Format optional benchmark numbers for CLI output."""

    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _print_tagger_summary(result: TaggerBenchmarkResult, output_json: Path) -> None:
    """Print a compact benchmark summary."""

    print("Tagger benchmark")
    print(f"  tagger: {result.tagger}")
    print(
        f"  selected/processed/measured/failed: {result.selected_images}/{result.processed_images}/"
        f"{result.measured_images}/{result.failed_images}"
    )
    print(f"  batch size / warmup batches: {result.batch_size} / {result.warmup_batches}")
    print(f"  providers available: {', '.join(result.providers_available) or 'n/a'}")
    print(f"  providers session: {', '.join(result.providers_session) or 'n/a'}")
    print(f"  total seconds: {result.total_seconds:.3f}")
    print(f"  measured seconds: {result.measured_seconds:.3f}")
    print(f"  images/sec: {result.images_per_second:.3f}")
    print(
        "  batch seconds mean/p50/p95: "
        f"{_fmt_optional(result.batch_seconds_mean)} / "
        f"{_fmt_optional(result.batch_seconds_p50)} / "
        f"{_fmt_optional(result.batch_seconds_p95)}"
    )
    print(
        "  ort ms mean/p50/p95: "
        f"{_fmt_optional(result.ort_ms_mean)} / "
        f"{_fmt_optional(result.ort_ms_p50)} / "
        f"{_fmt_optional(result.ort_ms_p95)}"
    )
    print(
        "  post ms mean/p50/p95: "
        f"{_fmt_optional(result.post_ms_mean)} / "
        f"{_fmt_optional(result.post_ms_p50)} / "
        f"{_fmt_optional(result.post_ms_p95)}"
    )
    print(f"  json: {output_json}")


def _add_tagger_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the tagger benchmark subcommand."""

    parser = subparsers.add_parser("tagger", help="Benchmark ONNX tagger throughput")
    parser.add_argument("--root", type=Path, required=True, help="Image root or single image to benchmark")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of images to benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per prepared inference batch")
    parser.add_argument("--tagger", choices=("wd14", "pixai"), required=True, help="Tagger implementation")
    parser.add_argument("--model", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--tags-csv", type=Path, default=None, help="selected_tags.csv path")
    parser.add_argument("--warmup-batches", type=int, default=2, help="Initial batches excluded from summary metrics")
    parser.add_argument("--prefetch-batches", type=int, default=4, help="Prepared batch queue depth")
    parser.add_argument("--io-workers", type=int, default=None, help="Image loader worker count")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/bench"),
        help="Directory used with --auto-number",
    )
    parser.add_argument("--run-slug", default="tagging-baseline", help="Slug used with --auto-number")
    parser.add_argument(
        "--auto-number",
        action="store_true",
        help="Write JSON as NNN-<tagger>-<run-slug>.json under --output-dir",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write benchmark JSON",
    )
    parser.set_defaults(func=_run_tagger_command)


def _run_tagger_command(args: argparse.Namespace) -> int:
    """Execute the tagger benchmark command."""

    result = _benchmark_tagger(args)
    output_json = _resolve_output_json(args)
    _write_json(output_json, result)
    _print_tagger_summary(result, output_json)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_tagger_parser(subparsers)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
