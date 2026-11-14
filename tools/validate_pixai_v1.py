"""Validate a local PixAI v1 ONNX model and benchmark prepared/end-to-end inference.

Optional FP32 reference uses local upstream files and separately installed
Transformers/timm/Torch. No downloads or remote Python execution are performed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from tagger.base import TagCategory
from tagger.pixai_v1_identity import content_digest
from tagger.pixai_v1_labels import DEFAULT_THRESHOLDS, PREPROCESS_VERSION
from tagger.pixai_v1_onnx import PixaiV1OnnxTagger
from tagger.pixai_v1_preprocessing import composite_rgb


def _gpu_memory(samples: list[int], stop: threading.Event) -> None:
    """Sample total GPU memory; this includes other processes and is not an allocator peak."""
    while not stop.is_set():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            samples.append(int(output.splitlines()[0]))
        except (OSError, ValueError, subprocess.CalledProcessError):
            return
        stop.wait(0.1)


def _compare_reference(tagger: PixaiV1OnnxTagger, paths: list[Path], directory: Path) -> dict[str, object]:
    """Compare official FP32 model and processor, separating preprocessing and graph error."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True).eval().cuda()
    processor = AutoImageProcessor.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
    thresholds = np.array([DEFAULT_THRESHOLDS[TagCategory(label.category).name.lower()] for label in tagger._labels])
    results = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                original = image.copy()
            reference_input = processor(original, return_tensors="pt")["pixel_values"]
            prepared = tagger.prepare_batch_from_rgb_np([composite_rgb(original)])
            reference = torch.sigmoid(model(reference_input.cuda()).float()).cpu().numpy()
            graph = torch.sigmoid(torch.from_numpy(tagger._run_logits(reference_input.numpy()))).numpy()
            actual = torch.sigmoid(torch.from_numpy(tagger._run_logits(prepared))).numpy()
            diff = np.abs(actual - reference)
            results.append(
                dict(
                    image=str(path),
                    input_max_abs=float(np.max(np.abs(prepared - reference_input.numpy()))),
                    graph_max_abs=float(np.max(np.abs(graph - reference))),
                    probability_max_abs=float(diff.max()),
                    probability_mean_abs=float(diff.mean()),
                    threshold_disagreements=int(np.count_nonzero((actual > thresholds) != (reference > thresholds))),
                    allclose=bool(np.allclose(actual, reference, atol=1e-3, rtol=1e-3)),
                )
            )
            print(f"reference: {path.name} max={diff.max():.6g}", flush=True)
    del model
    torch.cuda.empty_cache()
    return summarize_reference(results)


def summarize_reference(results: list[dict[str, object]]) -> dict[str, object]:
    """
    公式 FP32 互換の合否. allclose でも閾値をまたげば保存タグ集合が変わるため、
    数値一致(numerical_pass)と採用判定一致(decision_pass)の両方を要求する.
    """
    numerical = all(bool(item["allclose"]) for item in results)
    decision = all(item["threshold_disagreements"] == 0 for item in results)
    return dict(
        passed=numerical and decision,
        numerical_pass=numerical,
        decision_pass=decision,
        atol=1e-3,
        rtol=1e-3,
        images=results,
    )


def main() -> int:
    """Write model contract, benchmark and optional numerical comparison as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/pixai-v1/validation.json"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path("tmp/pixai-v1/hf-modules").resolve()))
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})[
        : args.limit
    ]
    if not paths:
        parser.error("No images found")
    tagger = PixaiV1OnnxTagger(
        args.model,
        args.labels_json,
        providers=["CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"],
    )
    samples: list[int] = []
    stop = threading.Event()
    monitor = threading.Thread(target=_gpu_memory, args=(samples, stop), daemon=True)
    monitor.start()
    import onnxruntime

    weights = args.model.with_name(args.model.name + ".data")
    report = dict(
        model=str(args.model),
        model_sha256=content_digest(args.model),
        external_weights_sha256=content_digest(weights) if weights.is_file() else None,
        labels_sha256=content_digest(tagger._labels_path),
        preprocess=PREPROCESS_VERSION,
        ort_version=onnxruntime.__version__,
        providers=tagger._session.get_providers(),
        input_shape=tagger._session.get_inputs()[0].shape,
        output_shape=tagger._session.get_outputs()[0].shape,
        benchmarks=[],
    )
    try:
        for size in [1, 2, 4, 8]:
            selected = [paths[i % len(paths)] for i in range(size)]
            started = perf_counter()
            images = []
            for path in selected:
                with Image.open(path) as image:
                    images.append(composite_rgb(image))
            prepared = tagger.prepare_batch_from_rgb_np(images)
            preprocessed = perf_counter()
            tagger.infer_batch_prepared(prepared)  # warmup
            inferred = perf_counter()
            tagger.infer_batch_prepared(prepared)
            elapsed = perf_counter() - inferred
            report["benchmarks"].append(
                dict(
                    requested_batch=size,
                    execution_batch=tagger._batch_limit,
                    prepared_images_per_second=size / elapsed,
                    end_to_end_images_per_second=size / (elapsed + preprocessed - started),
                )
            )
        if args.reference_dir:
            report["reference"] = _compare_reference(tagger, paths, args.reference_dir)
    finally:
        stop.set()
        monitor.join(timeout=5)
        tagger.close()
    report["max_sampled_total_gpu_memory_mib"] = max(samples) if samples else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 1 if args.reference_dir and not report["reference"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
