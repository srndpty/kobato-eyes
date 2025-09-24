"""ONNX Runtime implementation of the WD14 tagger."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from tagger.base import ITagger, MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap

try:  # pragma: no cover - import is environment dependent
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - graceful degradation
    ort = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


ONNXRUNTIME_MISSING_MESSAGE = "onnxruntime is required. Try: pip install onnxruntime-gpu  (or onnxruntime for CPU)"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class _Label:
    name: str
    category: TagCategory


_CATEGORY_LOOKUP = {
    "0": TagCategory.GENERAL,
    "general": TagCategory.GENERAL,
    "1": TagCategory.CHARACTER,
    "character": TagCategory.CHARACTER,
    "2": TagCategory.RATING,
    "rating": TagCategory.RATING,
    "3": TagCategory.COPYRIGHT,
    "copyright": TagCategory.COPYRIGHT,
    "4": TagCategory.ARTIST,
    "artist": TagCategory.ARTIST,
    "5": TagCategory.META,
    "meta": TagCategory.META,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


logger = logging.getLogger(__name__)

_DEFAULT_TAG_FILES = (
    "selected_tags.csv",
    "selected_tags_v3.csv",
    "selected_tags_v3c.csv",
)


def ensure_onnxruntime() -> None:
    """Ensure onnxruntime is importable, raising a user-facing error otherwise."""

    if ort is None:  # pragma: no cover - runtime guard
        raise RuntimeError(ONNXRUNTIME_MISSING_MESSAGE) from _IMPORT_ERROR


def get_available_providers() -> list[str]:
    """Return the list of ONNX Runtime providers available on this system."""

    ensure_onnxruntime()
    try:
        providers = list(ort.get_available_providers())  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("WD14: failed to query ONNX providers: %s", exc)
        return []
    return providers


class WD14Tagger(ITagger):
    """WD14 tagger backed by ONNX Runtime with CUDA acceleration when available."""

    def __init__(
        self,
        model_path: str | Path,
        labels_csv: str | Path | None = None,
        *,
        tags_csv: str | Path | None = None,
        providers: Iterable[str] | None = None,
        input_size: int = 448,
        default_thresholds: ThresholdMap | None = None,
        default_max_tags: MaxTagsMap | None = None,
    ) -> None:
        ensure_onnxruntime()

        self._model_path = Path(model_path)
        logger.info("WD14: using model %s", self._model_path)
        labels_path = self._resolve_labels_path(tags_csv or labels_csv)
        self._labels_path = labels_path
        self._labels = self._load_labels(labels_path)
        self._input_size = int(input_size)
        self._default_thresholds = dict(default_thresholds or {})
        self._default_max_tags = dict(default_max_tags or {})

        session_options = ort.SessionOptions()
        available_providers = get_available_providers()
        if available_providers:
            logger.info("WD14: available ONNX providers: %s", ", ".join(available_providers))
        else:
            logger.warning("WD14: no ONNX providers reported by runtime")

        requested_providers: Sequence[str] | None = providers
        provider_attempts: list[list[str]]
        if requested_providers is not None:
            provider_attempts = [list(requested_providers)]
            if _CUDA_PROVIDER in provider_attempts[0] and _CUDA_PROVIDER not in available_providers:
                logger.warning(
                    "WD14: %s requested but not available; falling back to %s",
                    _CUDA_PROVIDER,
                    _CPU_PROVIDER,
                )
                provider_attempts = [[_CPU_PROVIDER]]
        else:
            if available_providers and _CUDA_PROVIDER not in available_providers:
                logger.info(
                    "WD14: %s not reported by runtime; CPU provider will be used if CUDA fails",
                    _CUDA_PROVIDER,
                )
            provider_attempts = [[_CUDA_PROVIDER], [_CPU_PROVIDER]]
        last_error: Exception | None = None
        session = None
        chosen_providers: list[str] | None = None
        for provider_list in provider_attempts:
            try:
                session = ort.InferenceSession(
                    str(self._model_path),
                    sess_options=session_options,
                    providers=provider_list,
                )
            except Exception as exc:  # pragma: no cover - handled in tests via mocks
                last_error = exc
                if requested_providers is None and provider_list == [_CUDA_PROVIDER]:
                    logger.warning(
                        "WD14: %s unavailable, falling back to %s",
                        _CUDA_PROVIDER,
                        _CPU_PROVIDER,
                    )
                    continue
                raise
            else:
                chosen_providers = provider_list
                break
        if session is None or chosen_providers is None:
            raise RuntimeError("WD14: failed to initialise ONNX Runtime session") from last_error
        if requested_providers is None and chosen_providers == [_CUDA_PROVIDER]:
            logger.info("WD14: using %s", _CUDA_PROVIDER)
        elif requested_providers is None and chosen_providers == [_CPU_PROVIDER]:
            logger.info("WD14: using %s", _CPU_PROVIDER)
        else:
            logger.info("WD14: using providers %s", chosen_providers)
        self._session = session
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

        if len(self._output_names) != 1:
            raise RuntimeError("Expected a single output tensor from WD14 ONNX model, got " f"{self._output_names}")

    def _resolve_labels_path(self, explicit: str | Path | None) -> Path:
        if explicit is not None:
            path = Path(str(explicit)).expanduser()
            if path.is_file():
                logger.info("WD14: using tags CSV %s", path)
                return path
            message = f"WD14: tags CSV not found at {path}"
            logger.warning(message)
            raise FileNotFoundError(message)

        search_dir = self._model_path.parent
        candidates: list[Path] = []

        def add_candidate(candidate: Path) -> None:
            if candidate not in candidates:
                candidates.append(candidate)

        for name in _DEFAULT_TAG_FILES:
            add_candidate(search_dir / name)
        add_candidate(self._model_path.with_suffix(".csv"))
        for extra in sorted(search_dir.glob("selected_tags*.csv")):
            add_candidate(extra)

        for candidate in candidates:
            if candidate.is_file():
                logger.info("WD14: using tags CSV %s", candidate)
                return candidate

        message = "WD14: selected_tags.csv not found"
        logger.warning(message)
        raise FileNotFoundError(message)

    @staticmethod
    def _load_labels(labels_csv: str | Path) -> list[_Label]:
        labels: list[_Label] = []
        with Path(labels_csv).open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) == 1:
                    tag_name = row[0].strip()
                    category = TagCategory.GENERAL
                else:
                    tag_name = row[0].strip()
                    category_key = row[1].strip().lower()
                    category = _CATEGORY_LOOKUP.get(category_key, TagCategory.GENERAL)
                if tag_name:
                    labels.append(_Label(name=tag_name, category=category))
        if not labels:
            raise ValueError("No labels parsed from WD14 label CSV")
        return labels

    # original
    # def _preprocess(self, image: Image.Image) -> np.ndarray:
    #     rgb = image.convert("RGB")
    #     resample = getattr(Image, "Resampling", Image).BICUBIC  # type: ignore[attr-defined]
    #     resized = rgb.resize((self._input_size, self._input_size), resample)
    #     array = np.asarray(resized, dtype=np.float32) / 255.0  # 0..1
    #     # ★ BGR反転もしない、転置もしない → NHWC のまま
    #     return np.expand_dims(array, 0)  # (1, H, W, 3)

    def make_square(self, img, target_size):
        old_size = img.shape[:2]
        desired_size = max(old_size)
        desired_size = max(desired_size, target_size)

        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [255, 255, 255]
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # noinspection PyUnresolvedReferences
    def smart_resize(self, img, size):
        # Assumes the image has already gone through make_square
        if img.shape[0] > size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        elif img.shape[0] < size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        else:  # just do nothing
            pass

        return img

    # borrowed from: https://huggingface.co/spaces/deepghs/wd14_tagging_online/blob/main/app.py
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        _, height, _, _ = self._session.get_inputs()[0].shape

        # alpha to white
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = self.make_square(image, height)
        image = self.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)
        return image

    @staticmethod
    def _resolve_thresholds(
        defaults: dict[TagCategory, float], overrides: ThresholdMap | None
    ) -> dict[TagCategory, float]:
        merged = dict(defaults)
        if overrides:
            merged.update(overrides)
        return merged

    @staticmethod
    def _resolve_max_tags(defaults: dict[TagCategory, int], overrides: MaxTagsMap | None) -> dict[TagCategory, int]:
        merged = dict(defaults)
        if overrides:
            merged.update(overrides)
        return merged

    def infer_batch(
        self,
        images: Iterable[Image.Image],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        image_list = list(images)
        if not image_list:
            return []

        batch = np.vstack([self._preprocess(image) for image in image_list])
        outputs = self._session.run(self._output_names, {self._input_name: batch})
        logits = outputs[0]

        if logits.shape[1] != len(self._labels):
            raise RuntimeError(
                f"Model output dimension {logits.shape[1]} does not match label count {len(self._labels)}"
            )

        # probabilities = _sigmoid(logits)
        # Auto-detect: if the tensor already looks like probabilities in [0,1],
        # skip sigmoid; otherwise apply sigmoid to logits.
        minv = float(np.min(logits))
        maxv = float(np.max(logits))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("WD14 LOGITS: min=%.4f max=%.4f", minv, maxv)

        if 0.0 <= minv <= 1.0 and 0.0 <= maxv <= 1.0:
            probabilities = logits.astype(np.float32, copy=False)  # already probs
        else:
            probabilities = _sigmoid(logits)

        # --- DEBUG: dump raw top-k (before thresholds) for the 1st image in this batch ---
        # try:
        #     p0 = probabilities[0]
        #     topk_n = 20
        #     idx = np.argpartition(-p0, topk_n)[:topk_n]
        #     idx = idx[np.argsort(-p0[idx])]
        #     dump = []
        #     for i in idx:
        #         lab = self._labels[int(i)]
        #         dump.append(f"{lab.name}({lab.category.name})={p0[i]:.3f}")
        #     print("WD14 RAW top20: %s", ", ".join(dump))
        # except Exception as _e:  # 失敗しても推論自体は続行
        #     print("top-k debug dump failed: %r", _e)
        # --- DEBUG end ---

        resolved_thresholds = self._resolve_thresholds(self._default_thresholds, thresholds)
        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)

        results: list[TagResult] = []
        for probs in probabilities:
            predictions: list[TagPrediction] = []
            by_category: dict[TagCategory, list[TagPrediction]] = {}
            for label, score in zip(self._labels, probs):
                probability = float(score)
                threshold = resolved_thresholds.get(label.category, 0.0)
                if probability < threshold:
                    continue
                prediction = TagPrediction(name=label.name, score=probability, category=label.category)
                by_category.setdefault(label.category, []).append(prediction)

            for category, preds in by_category.items():
                preds.sort(key=lambda item: item.score, reverse=True)
                limit = resolved_limits.get(category)
                if limit is not None:
                    preds = preds[: max(limit, 0)]
                predictions.extend(preds)

            predictions.sort(key=lambda item: item.score, reverse=True)
            results.append(TagResult(tags=predictions))
        return results


__all__ = ["WD14Tagger", "ensure_onnxruntime", "get_available_providers", "ONNXRUNTIME_MISSING_MESSAGE"]
