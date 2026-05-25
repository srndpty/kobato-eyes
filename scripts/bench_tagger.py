"""Run a focused tagger throughput benchmark and write JSON metrics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config.schema import PipelineSettings, TaggerSettings  # noqa: E402
from core.pipeline.loaders import PrefetchLoaderPrepared  # noqa: E402
from core.pipeline.resolver import _resolve_tagger  # noqa: E402
from core.scanner import DEFAULT_EXTENSIONS, iter_images  # noqa: E402
from tagger.wd14_onnx import get_available_providers  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    """Return a percentile value from a non-empty sample."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return float(ordered[index])


def _float_list_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _build_settings(args: argparse.Namespace) -> PipelineSettings:
    tagger = TaggerSettings(
        name=args.tagger,
        provider=args.provider,
        device=args.device,
        model_path=str(args.model),
        tags_csv=str(args.tags_csv) if args.tags_csv else None,
    )
    return PipelineSettings(
        roots=[str(args.root)],
        allow_exts=set(args.extensions or sorted(DEFAULT_EXTENSIONS)),
        batch_size=int(args.batch_size),
        tagger=tagger,
    )


def _collect_paths(args: argparse.Namespace) -> list[str]:
    roots = [Path(args.root).expanduser()]
    excluded = [Path(path).expanduser() for path in args.exclude]
    extensions = args.extensions or sorted(DEFAULT_EXTENSIONS)
    paths = list(iter_images(roots, excluded=excluded, extensions=extensions))
    paths.sort(key=lambda path: (path.parent, path.stat().st_size if path.exists() else 0))
    if args.limit > 0:
        paths = paths[: args.limit]
    return [str(path) for path in paths]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the benchmark and return JSON-serialisable metrics."""

    if args.prefetch_depth is not None:
        os.environ["KE_PREFETCH_DEPTH"] = str(args.prefetch_depth)
    if args.io_workers is not None:
        os.environ["KE_IO_WORKERS"] = str(args.io_workers)
    if args.topk_cap is not None:
        os.environ["KE_PIXAI_TOPK_CAP"] = str(args.topk_cap)

    resolve_started = time.perf_counter()
    settings = _build_settings(args)
    tagger, thresholds, max_tags = _resolve_tagger(settings, None, thresholds=None, max_tags=None)
    resolve_seconds = time.perf_counter() - resolve_started
    scan_started = time.perf_counter()
    paths = _collect_paths(args)
    scan_seconds = time.perf_counter() - scan_started

    batches: list[dict[str, Any]] = []
    ort_ms_values: list[float] = []
    post_ms_values: list[float] = []
    processed = 0
    measured_images = 0
    measured_seconds = 0.0
    total_started = time.perf_counter()
    loader = PrefetchLoaderPrepared(
        paths,
        tagger=tagger,
        batch_size=int(args.batch_size),
        prefetch_batches=int(args.prefetch_depth or 4),
        io_workers=args.io_workers,
    )
    try:
        loader_iter = iter(loader)
        batch_index = 0
        while True:
            wait_started = time.perf_counter()
            try:
                batch_paths, np_batch, _sizes = next(loader_iter)
            except StopIteration:
                break
            batch_index += 1
            wait_batch_seconds = time.perf_counter() - wait_started
            infer_started = time.perf_counter()
            results = tagger.infer_batch_prepared(np_batch, thresholds=thresholds or None, max_tags=max_tags or None)
            infer_seconds = time.perf_counter() - infer_started
            infer_metrics = dict(getattr(tagger, "_last_infer_metrics", {}) or {})
            image_count = len(results)
            warmup = batch_index <= int(args.warmup_batches)
            batch_seconds = wait_batch_seconds + infer_seconds
            processed += image_count
            if not warmup:
                measured_images += image_count
                measured_seconds += batch_seconds
                if "ort_ms" in infer_metrics:
                    ort_ms_values.append(float(infer_metrics["ort_ms"]))
                if "post_ms" in infer_metrics:
                    post_ms_values.append(float(infer_metrics["post_ms"]))
            batches.append(
                {
                    "batch_index": batch_index,
                    "image_count": image_count,
                    "warmup": warmup,
                    "wait_batch_seconds": wait_batch_seconds,
                    "infer_seconds": infer_seconds,
                    "batch_seconds": batch_seconds,
                    "ort_ms": infer_metrics.get("ort_ms"),
                    "post_ms": infer_metrics.get("post_ms"),
                    "tagger_total_ms": infer_metrics.get("total_ms"),
                    "first_path": batch_paths[0] if batch_paths else None,
                }
            )
    finally:
        loader.close()
        closer = getattr(tagger, "close", None)
        if callable(closer):
            closer()

    total_seconds = time.perf_counter() - total_started
    measured_batches = [batch for batch in batches if not batch["warmup"]]
    batch_seconds = [float(batch["batch_seconds"]) for batch in measured_batches]
    infer_seconds = [float(batch["infer_seconds"]) for batch in measured_batches]
    wait_seconds = [float(batch["wait_batch_seconds"]) for batch in measured_batches]
    provider_session = []
    session = getattr(tagger, "_session", None)
    if session is not None and hasattr(session, "get_providers"):
        provider_session = list(session.get_providers())

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(Path(args.root).expanduser()),
        "tagger": args.provider,
        "model": str(Path(args.model).expanduser()),
        "tags_csv": str(Path(args.tags_csv).expanduser()) if args.tags_csv else None,
        "selected_images": len(paths),
        "processed_images": processed,
        "measured_images": measured_images,
        "batch_size": int(args.batch_size),
        "warmup_batches": int(args.warmup_batches),
        "prefetch_batches": int(args.prefetch_depth or 4),
        "io_workers": args.io_workers,
        "providers_available": get_available_providers(),
        "providers_session": provider_session,
        "total_seconds": total_seconds,
        "resolve_seconds": resolve_seconds,
        "scan_seconds": scan_seconds,
        "measured_seconds": measured_seconds,
        "images_per_second": measured_images / measured_seconds if measured_seconds > 0.0 else 0.0,
        "batch_seconds_mean": _float_list_mean(batch_seconds),
        "batch_seconds_p50": _percentile(batch_seconds, 0.50),
        "batch_seconds_p95": _percentile(batch_seconds, 0.95),
        "infer_seconds_mean": _float_list_mean(infer_seconds),
        "infer_seconds_p50": _percentile(infer_seconds, 0.50),
        "infer_seconds_p95": _percentile(infer_seconds, 0.95),
        "wait_batch_seconds_mean": _float_list_mean(wait_seconds),
        "wait_batch_seconds_p50": _percentile(wait_seconds, 0.50),
        "wait_batch_seconds_p95": _percentile(wait_seconds, 0.95),
        "ort_ms_mean": _float_list_mean(ort_ms_values),
        "ort_ms_p50": _percentile(ort_ms_values, 0.50),
        "ort_ms_p95": _percentile(ort_ms_values, 0.95),
        "post_ms_mean": _float_list_mean(post_ms_values),
        "post_ms_p50": _percentile(post_ms_values, 0.50),
        "post_ms_p95": _percentile(post_ms_values, 0.95),
        "loader_metrics": loader.metrics_snapshot().as_dict(),
        "db_write_seconds": 0.0,
        "fts_rebuild_seconds": 0.0,
        "batches": batches,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tags-csv", type=Path)
    parser.add_argument("--tagger", default="wd14-onnx")
    parser.add_argument("--provider", choices=["auto", "wd14", "pixai"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--prefetch-depth", type=int)
    parser.add_argument("--io-workers", type=int)
    parser.add_argument("--topk-cap", type=int)
    parser.add_argument("--extension", dest="extensions", action="append")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "tmp" / "bench" / "tagger-bench.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("processed_images", "images_per_second", "total_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
