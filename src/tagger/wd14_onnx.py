"""ONNX Runtime implementation of the WD14 tagger."""

from __future__ import annotations

import logging
import os
import weakref
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from tagger.base import ITagger, MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap
from tagger.labels_util import load_selected_tags

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


_ACTIVE_TAGGERS: "weakref.WeakSet[WD14Tagger]"  # type: ignore[name-defined]


@dataclass(frozen=True)
class _Label:
    name: str
    category: TagCategory
    count: int = 0


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
        _configure_session_options(session_options)
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
            print("WD14: using %s", _CUDA_PROVIDER)
        elif requested_providers is None and chosen_providers == [_CPU_PROVIDER]:
            print("WD14: using %s", _CPU_PROVIDER)
        else:
            print("WD14: using providers %s", chosen_providers)
        self._session = session
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

        # ==== 追加: ベクトル化用の前計算（名前/カテゴリ/カテゴリ→index配列） ====
        # Python ループ削減のため、一度だけ NumPy 配列化して保持
        self._label_names = np.array([lab.name for lab in self._labels], dtype=object)
        self._label_cats = np.fromiter(
            (int(lab.category) for lab in self._labels), dtype=np.int16, count=len(self._labels)
        )
        self._cat_to_idx: dict[int, np.ndarray] = {
            int(cat): np.nonzero(self._label_cats == int(cat))[0] for cat in sorted(set(self._label_cats.tolist()))
        }
        # デフォルトしきい値ベクトルはキャッシュしておく（可変のときは都度生成）
        self._default_thr_vec = self._build_threshold_vector(self._default_thresholds)

        _log_provider_details(self._session, chosen_providers)
        _ACTIVE_TAGGERS.add(self)

        if len(self._output_names) != 1:
            raise RuntimeError("Expected a single output tensor from WD14 ONNX model, got " f"{self._output_names}")

    # ==== 追加: しきい値ベクトルを作る ====
    def _build_threshold_vector(self, thresholds: dict[TagCategory | int, float]) -> np.ndarray:
        # 未指定カテゴリは 0.0（=無制限）
        vec = np.zeros((len(self._labels),), dtype=np.float32)
        for k, v in thresholds.items():
            cat = int(k)
            idx = self._cat_to_idx.get(cat)
            if idx is not None and len(idx) > 0:
                vec[idx] = float(v)
        return vec

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
        for tag in load_selected_tags(labels_csv):
            name = tag.name.strip()
            if not name:
                continue
            try:
                category = TagCategory(tag.category)
            except ValueError:
                category = TagCategory.GENERAL
            count = int(tag.count or 0)
            labels.append(_Label(name=name, category=category, count=count))
        if not labels:
            raise ValueError("No labels parsed from WD14 label CSV")
        return labels

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

    # --- 追加: 外部からも使える前処理ラッパー（1枚 → (1,H,W,3) float32 BGR） ---
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        return self._preprocess(image)

    # --- 追加: すでに前処理済みのテンソル群で推論する高速パス ---
    def infer_preprocessed(
        self,
        tensors: Iterable[np.ndarray],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        batch_list = list(tensors)
        if not batch_list:
            return []
        # (N,H,W,3) が来る前提。vstack して一度に推論。
        batch = np.vstack(batch_list)
        ort_start = perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: batch})
        ort_ms = (perf_counter() - ort_start) * 1000.0
        logits = outputs[0]
        if logits.shape[1] != len(self._labels):
            raise RuntimeError(
                f"Model output dimension {logits.shape[1]} does not match label count {len(self._labels)}"
            )
        # 以降は infer_batch と同じ後処理
        post_start = perf_counter()
        minv = float(np.min(logits))
        maxv = float(np.max(logits))
        if 0.0 <= minv <= 1.0 and 0.0 <= maxv <= 1.0:
            probabilities = logits.astype(np.float32, copy=False)
        else:
            probabilities = _sigmoid(logits)
        resolved_thresholds = self._resolve_thresholds(self._default_thresholds, thresholds)
        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)
        results: list[TagResult] = []
        for probs in probabilities:
            predictions: list[TagPrediction] = []
            by_category: dict[TagCategory, list[TagPrediction]] = {}
            for label, score in zip(self._labels, probs):
                p = float(score)
                thr = resolved_thresholds.get(label.category, 0.0)
                if p < thr:
                    continue
                pred = TagPrediction(name=label.name, score=p, category=label.category)
                by_category.setdefault(label.category, []).append(pred)
            for cat, preds in by_category.items():
                preds.sort(key=lambda it: it.score, reverse=True)
                limit = resolved_limits.get(cat)
                if limit is not None:
                    preds = preds[: max(limit, 0)]
                predictions.extend(preds)
            predictions.sort(key=lambda it: it.score, reverse=True)
            results.append(TagResult(tags=predictions))
        post_ms = (perf_counter() - post_start) * 1000.0
        batch_size = len(batch_list)
        imgs_per_second = batch_size / ((ort_ms + post_ms) / 1000.0) if (ort_ms + post_ms) > 0 else float("inf")
        logger.info(
            "WD14 prebatched batch=%d ort=%.2fms post=%.2fms imgs/s=%.2f", batch_size, ort_ms, post_ms, imgs_per_second
        )
        return results

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

        total_start = perf_counter()
        preprocess_start = perf_counter()
        batch = np.vstack([self._preprocess(image) for image in image_list])
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        ort_start = perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: batch})
        ort_ms = (perf_counter() - ort_start) * 1000.0
        logits = outputs[0]

        if logits.shape[1] != len(self._labels):
            raise RuntimeError(
                f"Model output dimension {logits.shape[1]} does not match label count {len(self._labels)}"
            )

        post_start = perf_counter()
        # --- ベクトル化: ロジット→確率（またはそのまま） ---
        minv = float(np.min(logits))
        maxv = float(np.max(logits))
        if 0.0 <= minv <= 1.0 and 0.0 <= maxv <= 1.0:
            probs = logits.astype(np.float32, copy=False)
        else:
            probs = _sigmoid(logits).astype(np.float32, copy=False)

        # しきい値: 辞書を解決 → ベクトル（既定と同じならキャッシュを再利用）
        resolved_thresholds = self._resolve_thresholds(self._default_thresholds, thresholds)
        thr_vec = (
            self._default_thr_vec
            if resolved_thresholds == self._default_thresholds
            else self._build_threshold_vector(resolved_thresholds)
        )
        # マスク（B,C）
        mask = probs >= thr_vec  # broadcast

        # スコア降順の index を一括で取る（B,C）
        # マスク外は -inf に落として並べ替え対象から外す
        masked_scores = np.where(mask, probs, -np.inf)
        order = np.argsort(-masked_scores, axis=1, kind="stable")  # 降順

        # カテゴリごとの上限
        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)

        results: list[TagResult] = []
        # 画像ごとに「並び済みインデックス」を舐めて、カテゴリ上限を満たす分だけ収集
        # しきい値通過していない要素は -inf になっているので break
        for b in range(order.shape[0]):
            picks_idx: list[int] = []
            picks_score: list[float] = []
            # カテゴリ別の残数（dictより固定長の配列が速い）
            remaining = np.full(8, np.iinfo(np.int32).max, dtype=np.int32)  # 0..7 くらいを想定
            for cat, lim in resolved_limits.items():
                if lim is not None:
                    c = int(cat)
                    if 0 <= c < remaining.size:
                        remaining[c] = max(0, int(lim))
            # 並び済みを先頭から見る（高スコア順）
            row = order[b]
            scores_b = masked_scores[b]
            cats = self._label_cats
            for j in row:
                s = scores_b[j]
                if not np.isfinite(s):  # -inf になったら以降は全部 -inf
                    break
                c = cats[j]
                if remaining[c] <= 0:
                    continue
                picks_idx.append(int(j))
                picks_score.append(float(s))
                remaining[c] -= 1
            # 最後に、選ばれたものを TagResult に詰める
            # ここだけ軽い Python 生成だが、数はしきい値通過分だけ
            preds = [
                TagPrediction(
                    name=str(self._label_names[i]),
                    score=picks_score[k],
                    category=TagCategory(int(self._label_cats[i])),
                )
                for k, i in enumerate(picks_idx)
            ]
            results.append(TagResult(tags=preds))

        post_ms = (perf_counter() - post_start) * 1000.0
        total_ms = (perf_counter() - total_start) * 1000.0

        batch_size = len(image_list)
        imgs_per_second = batch_size / (total_ms / 1000.0) if total_ms > 0.0 else float("inf")
        logger.info(
            "WD14 batch=%d preprocess=%.2fms ort=%.2fms post=%.2fms total=%.2fms imgs/s=%.2f",
            batch_size,
            preprocess_ms,
            ort_ms,
            post_ms,
            total_ms,
            imgs_per_second,
        )
        return results

    def end_profile(self) -> str | None:
        """Finish ONNX Runtime profiling for this tagger instance."""

        session = getattr(self, "_session", None)
        if session is None:
            return None
        if not hasattr(session, "end_profiling"):
            return None
        try:
            profile_path = session.end_profiling()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("WD14: failed to finalise profiling: %s", exc)
            return None
        if profile_path:
            logger.info("WD14: profiling saved to %s", profile_path)
        return profile_path


def end_all_profiles() -> None:
    """Invoke :meth:`WD14Tagger.end_profile` for all live tagger instances."""

    for tagger in list(_ACTIVE_TAGGERS):
        try:
            tagger.end_profile()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("WD14: failed to end profiling for %s", tagger)


def _configure_session_options(options: "ort.SessionOptions") -> None:
    """Apply default optimisation, logging, and profiling settings."""

    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, "ORT_ENABLE_ALL", 99)
    options.enable_profiling = False  # 必要に応じてTrueに
    options.log_severity_level = 2
    profile_dir = _resolve_profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)
    options.profile_file_prefix = str(profile_dir / "wd14")
    logger.info("WD14: profiling enabled (prefix=%s)", options.profile_file_prefix)


def _log_provider_details(session: "ort.InferenceSession", chosen: Sequence[str]) -> None:
    """Log provider details for diagnostics in a consistent format."""

    try:
        session_providers = list(session.get_providers())
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("WD14: failed to query session providers: %s", exc)
        session_providers = []
    try:
        provider_options = session.get_provider_options()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("WD14: failed to query provider options: %s", exc)
        provider_options = {}
    logger.info(
        "WD14 providers=%s session_providers=%s options=%s",
        list(chosen),
        session_providers,
        provider_options,
    )


def _resolve_profile_dir() -> Path:
    """Resolve the directory used for ONNX Runtime profile output files."""

    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "kobato-eyes" / "logs"
    return Path.home() / "kobato-eyes" / "logs"


_ACTIVE_TAGGERS = weakref.WeakSet()


__all__ = [
    "WD14Tagger",
    "end_all_profiles",
    "ensure_onnxruntime",
    "get_available_providers",
    "ONNXRUNTIME_MISSING_MESSAGE",
]
