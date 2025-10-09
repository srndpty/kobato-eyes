"""ONNX Runtime implementation of the WD14 tagger."""

from __future__ import annotations

import logging
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image

from tagger.base import ITagger, MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap
from tagger.labels_util import load_selected_tags

try:
    import torch

    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))  # Windows 3.8+
except Exception:
    pass

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

        # ==== 並列前処理の設定 ====
        try:
            _cpu = os.cpu_count() or 4
        except Exception:
            _cpu = 4
        self._pre_workers = int(os.getenv("KE_PREPROC_WORKERS", str(min(8, _cpu))))
        # OpenCV の内部スレッドが暴れないように抑制（必要なら環境変数で上書き）
        try:
            cv2.setNumThreads(int(os.getenv("KE_CV2_THREADS", "1")))
        except Exception:
            pass
        self._pre_exec: ThreadPoolExecutor | None = None
        if self._pre_workers > 1:
            # 注意: 終了時のクリーンアップは弱参照の finalizer やプロセス終了で開放される前提
            self._pre_exec = ThreadPoolExecutor(max_workers=self._pre_workers, thread_name_prefix="ke-pre")

        self._topk_cap = 128  # 上限256個まで
        self._score_floor = float(os.getenv("KE_TAG_SCORE_FLOOR", "0.1"))
        self._batch_seq = 0
        self._last_batch_end = None

        if len(self._output_names) != 1:
            raise RuntimeError("Expected a single output tensor from WD14 ONNX model, got " f"{self._output_names}")

    @property
    def input_size_px(self) -> int:
        return int(self._input_size)

    # ---- 追加: BGR(0..255 or uint8) のリストをまとめてモデル入力に整形 ----
    def prepare_batch_from_bgr(self, bgr_list: List[np.ndarray]) -> np.ndarray:
        H = self._session.get_inputs()[0].shape[1] or self._input_size
        out = np.empty((len(bgr_list), H, H, 3), dtype=np.float32)
        for i, im in enumerate(bgr_list):
            if im is None:
                # ダミーで白
                out[i] = 255.0
                continue
            # 4ch(PNG)なら白に合成
            if im.ndim == 3 and im.shape[2] == 4:
                bgr = im[:, :, :3].astype(np.float32)
                a = im[:, :, 3:4].astype(np.float32) / 255.0
                bgr = bgr * a + 255.0 * (1.0 - a)
                bgr = bgr.astype(np.uint8)
            else:
                bgr = im

            # 正方形パディング
            h, w = bgr.shape[:2]
            side = max(h, w, H)
            top = (side - h) // 2
            left = (side - w) // 2
            sq = cv2.copyMakeBorder(
                bgr, top, side - h - top, left, side - w - left, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

            # resize（元と比べて拡大: CUBIC / 縮小: AREA）
            interp = cv2.INTER_CUBIC if side < H else cv2.INTER_AREA
            resized = cv2.resize(sq, (H, H), interpolation=interp)

            out[i] = resized.astype(np.float32)  # 正規化は元実装に合わせず(=そのまま0..255)
        return out  # (B,H,W,3) float32

    def prepare_batch_pil(self, images: list[Image.Image]) -> np.ndarray:
        _, H, _, _ = self._session.get_inputs()[0].shape
        batch = np.vstack([self._preprocess(im) for im in images])  # (B,H,W,3) float32
        return batch

    def prepare_batch_from_rgb_np(self, imgs_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """
        Loader から受け取った RGB uint8 の np.ndarray 群を、
        既存 _preprocess(PIL) と完全同一ロジックで (B,H,W,3) float32 に整形する。
        """
        _, height, _, _ = self._session.get_inputs()[0].shape  # NHWC 前提
        B = len(imgs_rgb)
        out = np.empty((B, height, height, 3), dtype=np.float32)

        for i, arr in enumerate(imgs_rgb):
            if arr is None:
                # ダミー（全白）で埋める／もしくは raise にする
                out[i].fill(255.0)
                continue

            # arr: RGB uint8 期待。例外ケースに備えた防御
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)  # (H,W) -> (H,W,3)
            elif arr.shape[2] == 4:
                # RGBA -> white 合成（PIL と同等）
                # arr は RGBA（ここでは RGB 前提だが RGBA が来た場合も救済）
                rgb = arr[:, :, :3].astype(np.float32)
                alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
                rgb = rgb * alpha + 255.0 * (1.0 - alpha)
                arr = rgb.astype(np.uint8)
            else:
                # (H,W,3) のはず
                pass

            # PIL 経路では最後に「RGB -> BGR」してから正方パディング＆リサイズ
            bgr = arr[:, :, ::-1]  # RGB -> BGR

            bgr = self.make_square(bgr, height)
            bgr = self.smart_resize(bgr, height)

            out[i] = bgr.astype(np.float32)  # 0..255 の float32（既存と同じ）

        return out

    # ---- 追加: すでに整形済みバッチを推論 ----
    def infer_batch_prepared(
        self,
        batch_bgr_or_rgb_prepared: np.ndarray,
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        """
        すでに prepare_batch_* で (B,H,W,3) float32 に整形済みのバッチを受け取る。
        """
        start = perf_counter()
        idle_ms = 0.0
        if self._last_batch_end is not None:
            idle_ms = (start - self._last_batch_end) * 1000.0

        batch = np.ascontiguousarray(batch_bgr_or_rgb_prepared, dtype=np.float32)
        ort_start = perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: batch})
        ort_ms = (perf_counter() - ort_start) * 1000.0
        logits = outputs[0]
        # 以降は従来 infer_batch() の後半（post）と同じ処理へ
        post_start = perf_counter()
        results = self._postprocess_logits_topk(logits=logits, thresholds=thresholds, max_tags=max_tags)
        # results = self._postprocess_logits(logits, thresholds, max_tags)
        post_ms = (perf_counter() - post_start) * 1000.0

        # ログ（従来と同じ書式に揃える）
        batch_size = logits.shape[0]
        total_ms = (0.0) + ort_ms + post_ms
        imgs_per_second = batch_size / (total_ms / 1000.0) if total_ms > 0.0 else float("inf")

        self._last_batch_end = perf_counter()
        self._batch_seq += 1

        logger.info(
            "WD14 infer_batch_prepared batch=%d idle=%.2fms ort=%.2fms post=%.2fms total=%.2fms imgs/s=%.2f",
            batch_size,
            idle_ms,
            ort_ms,
            post_ms,
            total_ms,
            imgs_per_second,
        )

        return results

    # ---- 既存infer_batchの後半を切り出し（postだけ再利用）----
    def _postprocess_logits(
        self, logits: np.ndarray, thresholds: ThresholdMap | None, max_tags: MaxTagsMap | None
    ) -> list[TagResult]:
        if logits.shape[1] != len(self._labels):
            raise RuntimeError(f"Model output dim {logits.shape[1]} != labels {len(self._labels)}")

        post_start = perf_counter()
        minv, maxv = float(np.min(logits)), float(np.max(logits))
        if 0.0 <= minv <= 1.0 and 0.0 <= maxv <= 1.0:
            probs = logits.astype(np.float32, copy=False)
        else:
            probs = _sigmoid(logits).astype(np.float32, copy=False)

        resolved_thresholds = self._resolve_thresholds(self._default_thresholds, thresholds)
        thr_vec = (
            self._default_thr_vec
            if resolved_thresholds == self._default_thresholds
            else self._build_threshold_vector(resolved_thresholds)
        )
        # ★ 全カテゴリ共通の下限（例: 0.1）を合流
        floor = getattr(self, "_score_floor", 0.0)
        if floor > 0.0:
            np.maximum(thr_vec, floor, out=thr_vec)  # in-place で底上げ

        mask = probs >= thr_vec
        masked = np.where(mask, probs, -np.inf)
        order = np.argsort(-masked, axis=1, kind="stable")

        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)
        results: list[TagResult] = []
        cats = self._label_cats

        for b in range(order.shape[0]):
            picks_idx, picks_score = [], []
            remaining = np.full(8, np.iinfo(np.int32).max, dtype=np.int32)
            for cat, lim in resolved_limits.items():
                if lim is not None:
                    c = int(cat)
                    if 0 <= c < remaining.size:
                        remaining[c] = max(0, int(lim))
            row = order[b]
            srow = masked[b]
            for j in row:
                s = srow[j]
                if not np.isfinite(s):
                    break
                c = cats[j]
                if remaining[c] <= 0:
                    continue
                picks_idx.append(int(j))
                picks_score.append(float(s))
                remaining[c] -= 1

            preds = [
                TagPrediction(
                    name=str(self._label_names[i]), score=picks_score[k], category=TagCategory(int(self._label_cats[i]))
                )
                for k, i in enumerate(picks_idx)
            ]
            results.append(TagResult(tags=preds))
        post_ms = (perf_counter() - post_start) * 1000.0
        logger.info("WD14 post=%.2fms", post_ms)
        return results

    def _postprocess_logits_topk(
        self,
        logits: np.ndarray,
        *,
        thresholds: ThresholdMap | None,
        max_tags: MaxTagsMap | None,
    ) -> list[TagResult]:
        """
        logits → TagResult の高速後処理。
        - 閾値でマスク → top-K だけ argpartition → その小さな集合だけを降順ソート
        - カテゴリ上限を守りつつ高スコア順に採用
        """
        # --- 1) 確率化（モデルにより logit/確率の両方がある）
        if logits.dtype != np.float32:
            logits = logits.astype(np.float32, copy=False)
        mn, mx = float(np.min(logits)), float(np.max(logits))
        probs = logits if (0.0 <= mn <= 1.0 and 0.0 <= mx <= 1.0) else _sigmoid(logits).astype(np.float32, copy=False)

        B, C = probs.shape

        # --- 2) 閾値ベクトル（デフォルトキャッシュを再利用）
        resolved_thr = self._resolve_thresholds(self._default_thresholds, thresholds)
        thr_vec = (
            self._default_thr_vec
            if resolved_thr == self._default_thresholds
            else self._build_threshold_vector(resolved_thr)
        )

        # --- 3) カテゴリ上限
        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)

        # unbounded（None）を含むかで、画像ごとの K を決める
        has_unbounded = any(v is None for v in resolved_limits.values())
        base_cap = None if has_unbounded else max(sum(int(v) for v in resolved_limits.values() if v is not None), 64)
        hard_cap = max(1, int(self._topk_cap))

        names = self._label_names
        cats = self._label_cats  # (C,)

        results: list[TagResult] = []

        # --- 画像ごと（バッチ次元）に処理
        for b in range(B):
            scores = probs[b]  # (C,)
            # 閾値マスク
            hit_mask = scores >= thr_vec  # (C,)
            hit_count = int(hit_mask.sum())
            if hit_count == 0:
                results.append(TagResult(tags=[]))
                continue

            # 画像ごとの上限 K（大きすぎて全件ソートしないように）
            if base_cap is None:
                K = min(hit_count, hard_cap)  # unbounded がある時はヒット数か hard_cap の小さい方
            else:
                K = min(hit_count, base_cap, hard_cap)

            # -inf でマスク（閾値未満は候補から除外）
            masked = np.where(hit_mask, scores, -np.inf)

            # top-K 候補だけ取り出す（argpartition は O(C)）
            # すべて -inf の場合に備え、kth を安全側でクリップ
            kth = max(0, min(K - 1, C - 1))
            cand_idx = np.argpartition(-masked, kth)[:K]  # (≤K,)

            # その小さな集合だけ降順ソート（O(K log K)）
            cand_sorted = cand_idx[np.argsort(-masked[cand_idx], kind="stable")]

            # カテゴリ上限の残数（0..7 あたりまで想定、足りなければ自動拡張でもOK）
            remaining = np.full(8, np.iinfo(np.int32).max, dtype=np.int32)
            for cat, lim in resolved_limits.items():
                if lim is not None:
                    c = int(cat)
                    if 0 <= c < remaining.size:
                        remaining[c] = max(0, int(lim))

            picks_idx: list[int] = []
            picks_score: list[float] = []
            for j in cand_sorted:
                c = int(cats[j])
                if c < remaining.size and remaining[c] <= 0:
                    continue
                s = float(scores[j])
                if not np.isfinite(s) or s < float(thr_vec[j]):  # 念のための防御
                    continue
                picks_idx.append(int(j))
                picks_score.append(s)
                if c < remaining.size:
                    remaining[c] -= 1

            # TagResult へ（名前/カテゴリはベクトルから引くので高速）
            preds = [
                TagPrediction(
                    name=str(names[i]),
                    score=picks_score[k],
                    category=TagCategory(int(cats[i])),
                )
                for k, i in enumerate(picks_idx)
            ]
            results.append(TagResult(tags=preds))

        return results

    # 既存の _preprocess（1枚→(1,H,W,3)）は残しつつ、
    # バッチ用の「1枚→(H,W,3) float32」を返す軽量版を追加
    def _preprocess_np(self, image: Image.Image) -> np.ndarray:
        """1枚ぶんを (H, W, 3) float32 (BGR, 0..255) で返す。"""
        height = self._input_size

        # alpha to white（PillowはC実装でGIL解放）
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        rgb = np.asarray(image)  # HWC, uint8
        bgr = rgb[:, :, ::-1]  # RGB -> BGR

        bgr = self.make_square(bgr, height)
        bgr = self.smart_resize(bgr, height)
        # ここで float32 化。正規化はモデル仕様に合わせ 0..255 のまま（従来どおり）
        return bgr.astype(np.float32, copy=False)

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
        # ---------- 並列前処理 ----------
        preprocess_start = perf_counter()
        B = len(image_list)
        H = self._input_size
        # 事前確保した一塊のバッファ（再利用はタグ付け側のバッチライフサイクル次第）
        batch = np.empty((B, H, H, 3), dtype=np.float32)

        if self._pre_exec is None or self._pre_workers <= 1 or B == 1:
            # シングルスレッド（既定動作と同じ）
            for i, im in enumerate(image_list):
                batch[i] = self._preprocess(im)
                # batch[i] = self._preprocess_np(im)
        else:
            # 並列（Pillow/CV2 はC実装でGIL解放するので効果あり）
            # enumerate を渡して index を保つ
            def _work(pair):
                i, im = pair
                batch[i] = self._preprocess(im)
                # batch[i] = self._preprocess_np(im)

            # map は順序維持、各タスクはC実装で実行
            self._pre_exec.map(_work, enumerate(image_list), chunksize=max(1, B // (self._pre_workers * 2)))
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        # ---------- 推論 ----------
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
        floor = getattr(self, "_score_floor", 0.0)
        if floor > 0.0:
            np.maximum(thr_vec, floor, out=thr_vec)
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
    import os

    options.enable_profiling = bool(int(os.getenv("KE_ORT_PROFILE", "0")))
    options.log_severity_level = 2
    profile_dir = _resolve_profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)
    options.profile_file_prefix = str(profile_dir / "wd14")
    if options.enable_profiling:
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
