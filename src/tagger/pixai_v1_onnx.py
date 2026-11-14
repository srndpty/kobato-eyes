"""PixAI v1 ONNX inference with explicit logits and six-category labels."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from tagger.base import MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap
from tagger.onnx_backend import onnx_provider_options, plan_provider_attempts
from tagger.pixai_v1_labels import (
    DEFAULT_THRESHOLDS,
    LABEL_COUNT,
    PREPROCESS_VERSION,
    load_labels_json,
    resolve_labels_json,
)
from tagger.pixai_v1_preprocessing import INPUT_SIZE, composite_rgb, prepare_rgb_batch

logger = logging.getLogger(__name__)

# 検証済みプロファイル(noaione 版)の出力名. sigmoid 済み "probs" 等を受理すると二重 sigmoid になるため、
# 出力の意味を名前で固定する. 別配布物は検証済みプロファイルを追加するまで拒否する.
SUPPORTED_OUTPUT_NAMES = frozenset({"logits"})


def validate_session(session: Any) -> tuple[str, str, int | None]:
    """Validate the v1 tensor contract and return input/output names and batch."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("PixAI v1 requires one input and one logits output")
    image, logits = inputs[0], outputs[0]
    if image.type != "tensor(float)" or len(image.shape) != 4 or list(image.shape[1:]) != [3, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError(f"PixAI v1 requires float32 NCHW input at {INPUT_SIZE}px: {image.shape}")
    if len(logits.shape) != 2 or logits.shape[1] != LABEL_COUNT or logits.type != "tensor(float)":
        raise ValueError(f"PixAI v1 requires float32 logits [batch, {LABEL_COUNT}]: {logits.shape}")
    if logits.name not in SUPPORTED_OUTPUT_NAMES:
        raise ValueError(
            f"PixAI v1 output {logits.name!r} is not a verified logits output "
            f"(expected one of {sorted(SUPPORTED_OUTPUT_NAMES)})"
        )
    fixed = image.shape[0] if isinstance(image.shape[0], int) else None
    if fixed is not None and fixed < 1:
        raise ValueError("PixAI v1 input batch must be positive")
    if isinstance(logits.shape[0], int) and fixed != logits.shape[0]:
        raise ValueError("PixAI v1 input/output batch dimensions disagree")
    return image.name, logits.name, fixed


class PixaiV1OnnxTagger:
    """Local ONNX tagger; heavy work runs on the existing pipeline worker."""

    provider = "pixai-v1"
    input_size_px = INPUT_SIZE
    preserve_source_rgb = True
    preprocess_version = PREPROCESS_VERSION

    def __init__(
        self,
        model_path: str | Path,
        labels_json: str | Path | None = None,
        *,
        providers: Iterable[str] | None = None,
        max_batch: int | None = None,
    ) -> None:
        # Reuse the existing runtime import/DLL setup without adding Torch as a runtime dependency.
        from tagger import wd14_onnx
        from tagger.pixai_v1_identity import v1_content_identity

        wd14_onnx.ensure_onnxruntime()
        ort = wd14_onnx.ort
        self._model_path = Path(model_path).expanduser()
        if not self._model_path.is_file():
            raise FileNotFoundError(f"PixAI v1 model not found: {self._model_path}")
        self._labels_path = resolve_labels_json(self._model_path, labels_json)
        # セッション生成前後で内容が一致することを確認し、署名と実ロード内容の TOCTOU を塞ぐ.
        identity_before = v1_content_identity(self._model_path, self._labels_path)
        self._labels = load_labels_json(self._labels_path)
        self._session: Any = None
        requested = list(providers) if providers is not None else None
        plan = plan_provider_attempts(requested, ort.get_available_providers())
        for message in (*plan.infos, *plan.warnings):
            logger.info("PixAI v1: %s", message)
        if requested == ["CUDAExecutionProvider"] and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("PixAI v1: CUDA was requested but is unavailable")
        for index, attempt in enumerate(plan.attempts):
            try:
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                options.log_severity_level = 3
                provider_options = onnx_provider_options(attempt) or [{} for _ in attempt]
                if provider_options is not None:
                    for provider, config in zip(attempt, provider_options):
                        if provider == "CUDAExecutionProvider":
                            config["use_tf32"] = "0"
                self._session = ort.InferenceSession(
                    str(self._model_path),
                    sess_options=options,
                    providers=attempt,
                    provider_options=provider_options,
                )
                if (
                    requested == ["CUDAExecutionProvider"]
                    and "CUDAExecutionProvider" not in self._session.get_providers()
                ):
                    raise RuntimeError("PixAI v1: CUDA session fell back to CPU")
                break
            except Exception:
                if index == len(plan.attempts) - 1:
                    raise
                logger.warning("PixAI v1: provider %s failed; trying next provider", attempt, exc_info=True)
        if v1_content_identity(self._model_path, self._labels_path) != identity_before:
            self._session = None
            raise RuntimeError(
                "PixAI v1 model, weights or labels changed while loading; retry after the copy completes"
            )
        self.content_identity: dict[str, str] = identity_before
        self._input_name, self._output_name, self._fixed_batch = validate_session(self._session)
        # 動的バッチは要求バッチ上限から開始し、OOM 時のみ半減する.
        self._batch_limit = self._fixed_batch or max(1, int(max_batch or 1))
        self._last_infer_metrics: dict[str, float] = {}
        logger.info(
            "PixAI v1: model=%s labels=%s providers=%s",
            self._model_path,
            self._labels_path,
            self._session.get_providers(),
        )

    def prepare_batch_from_rgb_np(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Prepare RGB images with the official resize/pad semantics."""
        return prepare_rgb_batch(images)

    def _run_logits(self, batch: np.ndarray) -> np.ndarray:
        """Split fixed batches, pad tails, and halve dynamic batches on GPU OOM."""
        chunks = []
        position = 0
        while position < len(batch):
            count = min(self._batch_limit, len(batch) - position)
            inputs = batch[position : position + count]
            if self._fixed_batch and count < self._fixed_batch:
                inputs = np.concatenate([inputs, np.repeat(inputs[-1:], self._fixed_batch - count, axis=0)])
            try:
                values = np.asarray(self._session.run([self._output_name], {self._input_name: inputs})[0])
            except Exception as exc:
                message = str(exc).lower()
                if (
                    self._fixed_batch
                    or count <= 1
                    or not any(
                        word in message for word in ("out of memory", "cudaerrormemoryallocation", "failed to allocate")
                    )
                ):
                    raise
                self._batch_limit = max(1, count // 2)
                logger.warning("PixAI v1: GPU OOM; retrying with batch %d", self._batch_limit)
                continue
            if values.shape != (len(inputs), LABEL_COUNT) or not np.isfinite(values).all():
                raise ValueError("PixAI v1 returned invalid or non-finite logits")
            chunks.append(values[:count])
            position += count
        return np.concatenate(chunks) if chunks else np.empty((0, LABEL_COUNT), dtype=np.float32)

    def infer_batch_prepared(
        self,
        batch: np.ndarray,
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        """Apply sigmoid exactly once, then category thresholds and optional limits."""
        if batch.dtype != np.float32 or batch.ndim != 4 or batch.shape[1:] != (3, INPUT_SIZE, INPUT_SIZE):
            raise ValueError("PixAI v1 requires a prepared float32 NCHW batch")
        if not np.isfinite(batch).all():
            raise ValueError("PixAI v1 input contains non-finite values")
        started = perf_counter()
        logits = self._run_logits(np.ascontiguousarray(batch))
        inferred = perf_counter()
        probabilities = np.exp(-np.abs(logits))
        probabilities = np.where(logits >= 0, 1.0 / (1.0 + probabilities), probabilities / (1.0 + probabilities))
        effective = {TagCategory[name.upper()]: value for name, value in DEFAULT_THRESHOLDS.items()}
        effective.update(thresholds or {})
        results = []
        for scores in probabilities:
            tags: list[TagPrediction] = []
            for category in effective:
                selected = [
                    i
                    for i, label in enumerate(self._labels)
                    if label.category == category and scores[i] > effective[category]
                ]
                selected.sort(key=lambda i: (-float(scores[i]), i))
                limit = (max_tags or {}).get(category)
                if limit is not None and limit > 0:
                    selected = selected[:limit]
                tags.extend(TagPrediction(self._labels[i].name, float(scores[i]), category) for i in selected)
            results.append(TagResult(tags))
        self._last_infer_metrics = dict(
            ort_ms=(inferred - started) * 1000,
            post_ms=(perf_counter() - inferred) * 1000,
            total_ms=(perf_counter() - started) * 1000,
        )
        return results

    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        """Composite PIL inputs and run the same prepared inference path."""
        return self.infer_batch_prepared(
            prepare_rgb_batch([composite_rgb(image) for image in images]), thresholds=thresholds, max_tags=max_tags
        )

    def close(self) -> None:
        """Release the ONNX session after worker completion."""
        self._session = None
