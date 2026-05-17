"""ONNX Runtime tagger for the PixAI label set with copyright enrichment."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from tagger.base import MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap
from tagger.labels_util import BROKEN_TAG_PREFIX, TagMeta, load_selected_tags
from tagger.onnx_backend import validate_label_count

from .wd14_onnx import WD14Tagger, _sigmoid

logger = logging.getLogger(__name__)


class PixaiOnnxTagger(WD14Tagger):
    """PixAI ONNX tagger that derives copyrights from character predictions."""

    def _resolve_output_names(self, output_names: list[str]) -> list[str]:
        """Select the PixAI prediction tensor from the available outputs."""

        if not output_names:
            raise RuntimeError("PixAI: model does not expose any outputs")

        preferred_order = ("prediction", "logits")
        for name in preferred_order:
            if name in output_names:
                if len(output_names) > 1:
                    logger.info("PixAI: selecting output '%s' from available tensors %s", name, output_names)
                return [name]

        if len(output_names) == 1:
            return output_names

        raise RuntimeError(f"PixAI: unable to determine prediction tensor from outputs {output_names}")

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
        super().__init__(
            model_path,
            labels_csv=labels_csv,
            tags_csv=tags_csv,
            providers=providers,
            input_size=input_size,
            default_thresholds=default_thresholds,
            default_max_tags=default_max_tags,
        )

        self.provider = "pixai"
        self._tag_meta_index = self._build_tag_meta_index(self._labels_path)
        self._label_name_cache = [str(name) for name in self._label_names.tolist()]
        self._label_cat_cache = [int(cat) for cat in self._label_cats.tolist()]
        # ★ PixAIのselected_tags.csvのcategoryを最優先で使う（無ければWD14側のfallback）
        self._effective_cats: list[int] = []
        _from_meta = 0
        for nm, fb in zip(self._label_name_cache, self._label_cat_cache):
            meta = self._tag_meta_index.get(nm)
            if meta is not None and isinstance(getattr(meta, "category", None), (int, np.integer)):
                self._effective_cats.append(int(meta.category))
                _from_meta += 1
            else:
                self._effective_cats.append(int(fb))
        try:
            import numpy as _np

            uniq, cnt = _np.unique(_np.array(self._effective_cats, dtype=_np.int32), return_counts=True)
            logger.info(
                "PixAI: categories source -> meta=%d, fallback=%d; dist=%s",
                _from_meta,
                len(self._effective_cats) - _from_meta,
                dict(zip(uniq.tolist(), cnt.tolist())),
            )
        except (TypeError, ValueError):
            # Failure policy: category distribution logging is diagnostic-only.
            pass

        # 追加: preprocess.json をロード
        model_dir = Path(model_path).parent
        pp_path = model_dir / "preprocess.json"
        self._pixai_pp = None
        if pp_path.exists():
            with pp_path.open("r", encoding="utf-8") as f:
                self._pixai_pp = json.load(f).get("stages", [])
            logger.info("PixAI: loaded preprocess.json (%s)", pp_path)
        else:
            logger.warning("PixAI: preprocess.json not found; using fallback transforms")
            self._pixai_pp = []  # 後述のフォールバックで処理

        self._try_fix_label_order_with_json()
        self._refresh_pixai_fast_metadata()

    def _try_fix_label_order_with_json(self) -> None:
        # 1) JSON の場所を推定 or 指定
        model_dir = Path(self._model_path).parent
        cand = [
            model_dir / "tags_v0.9_13k.json",
            model_dir / "pixai_tags.json",
            Path(os.getenv("KE_PIXAI_TAGS_JSON", "")),
        ]
        json_path = next((p for p in cand if p and p.exists()), None)
        if not json_path:
            logger.info("PixAI: tags JSON not found; skip order verification")
            return

        # 2) JSON の tag_map から「index -> name」の理想配列を作る
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tag_map = data.get("tag_map") or {}
        if not tag_map:
            logger.warning("PixAI: tag_map missing in %s", json_path)
            return

        # index は 0..N-1 の連番を想定。欠番や "" はプレースホルダに置き換え
        N = len(self._label_name_cache)
        expected: list[str | None] = [None] * N
        for name, idx in tag_map.items():
            if 0 <= int(idx) < N:
                expected[int(idx)] = name if name else f"{BROKEN_TAG_PREFIX}{idx}"

        # 欠落分はプレースホルダで埋める
        for i in range(N):
            if expected[i] is None:
                expected[i] = f"{BROKEN_TAG_PREFIX}{i}"

        # 3) 現在の配列とどれだけ食い違うかカウント
        mismatches = sum(1 for i in range(N) if expected[i] != self._label_name_cache[i])
        if mismatches == 0:
            logger.info("PixAI: label order matches JSON (%s)", json_path)
            return

        logger.warning("PixAI: label order mismatch detected: %d / %d differ; fixing", mismatches, N)

        # 4) 名前配列を JSON 準拠に差し替え
        self._label_name_cache = [str(name) for name in expected]

        # 5) カテゴリ配列を、読み込んだ meta（CSV）から名前で再構築
        new_cats: list[int] = []
        for nm in self._label_name_cache:
            meta = self._tag_meta_index.get(nm)
            if meta:
                new_cats.append(int(meta.category))
            else:
                # 見つからないものは General に倒す（後工程で閾値＆プレースホルダ除外が効く）
                new_cats.append(int(TagCategory.GENERAL))
        self._label_cat_cache = new_cats
        self._effective_cats = list(new_cats)

        # 分布を再ログ
        vals, cnts = np.unique(np.array(self._label_cat_cache, dtype=np.int32), return_counts=True)
        logger.info("PixAI labels category distribution (after fix): %s", {int(v): int(c) for v, c in zip(vals, cnts)})

    def _refresh_pixai_fast_metadata(self) -> None:
        """Build PixAI lookup arrays used by hot preprocessing/postprocessing paths."""

        self._pixai_label_names = np.array(self._label_name_cache, dtype=object)
        self._pixai_effective_cats = np.array(self._effective_cats, dtype=np.int16)
        self._pixai_name_to_idx = {name: idx for idx, name in enumerate(self._label_name_cache)}
        self._pixai_cat_to_idx = {
            int(cat): np.nonzero(self._pixai_effective_cats == int(cat))[0]
            for cat in sorted(set(self._pixai_effective_cats.tolist()))
        }
        self._pixai_default_thr_vec = self._build_pixai_threshold_vector(self._default_thresholds)
        mean, std = self._resolve_pixai_normalize()
        self._pixai_mean_chw = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self._pixai_std_chw = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def _resolve_pixai_normalize(self) -> tuple[Sequence[float], Sequence[float]]:
        """Return normalization values from PixAI preprocess metadata."""

        mean: Sequence[float] = [0.5, 0.5, 0.5]
        std: Sequence[float] = [0.5, 0.5, 0.5]
        for stage in self._pixai_pp or []:
            stage_type = str(stage.get("type") or stage.get("op") or "").lower()
            if stage_type == "normalize":
                mean = stage.get("mean", mean)
                std = stage.get("std", std)
        return mean, std

    def prepare_batch_from_rgb_np(self, rgb_list: Sequence[np.ndarray]) -> np.ndarray:
        assert len(rgb_list) >= 1, "rgb_list empty"
        target = int(self._input_size)
        batch = np.empty((len(rgb_list), 3, target, target), dtype=np.float32)
        mean = self._pixai_mean_chw
        std = self._pixai_std_chw

        for index, arr in enumerate(rgb_list):
            rgb = self._coerce_rgb_uint8(arr)
            h, w = rgb.shape[:2]
            scale = target / max(1, min(w, h))
            nw = max(target, int(round(w * scale)))
            nh = max(target, int(round(h * scale)))
            if (nw, nh) != (w, h):
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)

            left = max(0, (rgb.shape[1] - target) // 2)
            top = max(0, (rgb.shape[0] - target) // 2)
            cropped = rgb[top : top + target, left : left + target]
            if cropped.shape[0] != target or cropped.shape[1] != target:
                cropped = cv2.resize(cropped, (target, target), interpolation=cv2.INTER_CUBIC)

            chw = np.transpose(cropped, (2, 0, 1)).astype(np.float32, copy=False)
            chw *= 1.0 / 255.0
            batch[index] = (chw - mean) / std

        return np.ascontiguousarray(batch, dtype=np.float32)

    @staticmethod
    def _coerce_rgb_uint8(arr: np.ndarray) -> np.ndarray:
        """Return an RGB uint8 image array suitable for PixAI preprocessing."""

        rgb = np.asarray(arr)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.ndim == 3 and rgb.shape[2] == 4:
            base = rgb[:, :, :3].astype(np.float32)
            alpha = rgb[:, :, 3:4].astype(np.float32) / 255.0
            rgb = (base * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        elif rgb.ndim == 3 and rgb.shape[2] >= 3:
            rgb = rgb[:, :, :3]
        else:
            raise ValueError(f"PixAI: unsupported image array shape {getattr(rgb, 'shape', None)}")

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _build_tag_meta_index(labels_csv: Path) -> dict[str, TagMeta]:
        index: dict[str, TagMeta] = {}
        try:
            for meta in load_selected_tags(labels_csv):
                index[meta.name] = meta
        except (csv.Error, OSError, UnicodeError, ValueError) as exc:
            # Failure policy: PixAI metadata enriches categories/copyrights. A
            # malformed optional CSV should degrade to WD14 fallback labels.
            logger.warning("PixAI: failed to parse tag metadata from %s: %s", labels_csv, exc)
        else:
            return index
        return index

    def _build_pixai_threshold_vector(self, thresholds: Mapping[Any, float]) -> np.ndarray:
        """Build a per-label threshold vector using PixAI effective categories."""

        names_count = len(getattr(self, "_label_name_cache", []))
        vec = np.zeros((names_count,), dtype=np.float32)
        cat_to_idx: Mapping[int, np.ndarray] = getattr(self, "_pixai_cat_to_idx", {})
        for key, value in thresholds.items():
            cat = int(key)
            idx = cat_to_idx.get(cat)
            if idx is not None and len(idx) > 0:
                vec[idx] = float(value)
        return vec

    def _merge_copyrights(
        self, probs_by_name: Mapping[str, tuple[float, int | TagCategory]]
    ) -> dict[str, tuple[float, int]]:
        merged: dict[str, tuple[float, int]] = {
            name: (float(score), int(category)) for name, (score, category) in probs_by_name.items()
        }
        if not merged:
            return merged
        for name, (score, category_value) in list(merged.items()):
            try:
                category = TagCategory(int(category_value))
            except ValueError:
                continue
            if category != TagCategory.CHARACTER:
                continue
            meta = self._tag_meta_index.get(name)
            if not meta or not meta.ips:
                continue
            for ip_name in meta.ips:
                existing = merged.get(ip_name)
                if existing is None:
                    merged[ip_name] = (float(score), int(TagCategory.COPYRIGHT))
                    continue
                current_score, _ = existing
                best = float(score) if score > current_score else float(current_score)
                merged[ip_name] = (best, int(TagCategory.COPYRIGHT))
        return merged

    def _postprocess_logits_topk(
        self,
        logits: np.ndarray,
        *,
        thresholds: ThresholdMap | None,
        max_tags: MaxTagsMap | None,
    ) -> list[TagResult]:
        if logits.dtype != np.float32:
            logits = logits.astype(np.float32, copy=False)
        mn, mx = float(np.min(logits)), float(np.max(logits))
        probs = logits if (0.0 <= mn <= 1.0 and 0.0 <= mx <= 1.0) else _sigmoid(logits).astype(np.float32, copy=False)

        resolved_thr = self._resolve_thresholds(self._default_thresholds, thresholds)
        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)
        score_floor = float(getattr(self, "_score_floor", 0.0))

        cat_limits: dict[int, int | None] = {}
        for category in TagCategory:
            limit_value = resolved_limits.get(category)
            cat_limits[int(category)] = int(limit_value) if limit_value is not None else None

        names = getattr(self, "_pixai_label_names", np.array(self._label_name_cache, dtype=object))
        cats = getattr(self, "_pixai_effective_cats", np.array(self._effective_cats, dtype=np.int16))
        validate_label_count(int(logits.shape[1]), len(names), backend_name="PixAI")
        thr_vec = (
            self._pixai_default_thr_vec.copy()
            if resolved_thr == self._default_thresholds
            else self._build_pixai_threshold_vector(resolved_thr)
        )
        if score_floor > 0.0:
            np.maximum(thr_vec, score_floor, out=thr_vec)

        results: list[TagResult] = []
        hard_cap = max(1, int(os.getenv("KE_PIXAI_TOPK_CAP", str(getattr(self, "_topk_cap", 128)))))
        name_to_idx = getattr(self, "_pixai_name_to_idx", {})

        for row in probs:
            row_arr: np.ndarray = np.asarray(row, dtype=np.float32)
            # 先頭1枚だけ実インデックスと現在マッピング名をダンプ
            if not hasattr(self, "_dbg_top_once"):
                self._dbg_top_once = True
                first_row: np.ndarray = np.asarray(probs[0], dtype=np.float32)
                top_idx = np.argsort(first_row)[-10:][::-1]
                pairs = [(int(i), self._label_name_cache[int(i)], float(first_row[int(i)])) for i in top_idx]
                logger.info("PixAI TOP10 indices (current mapping): %s", pairs)

            hit_mask = row_arr >= thr_vec
            hit_count = int(hit_mask.sum())
            if hit_count == 0:
                results.append(TagResult(tags=[]))
                continue

            k = min(hit_count, hard_cap)
            masked = np.where(hit_mask, row_arr, -np.inf)
            cand_idx = np.argpartition(-masked, max(0, k - 1))[:k]
            cand_sorted = cand_idx[np.argsort(-masked[cand_idx], kind="stable")]

            merged: dict[str, tuple[float, int]] = {}
            for idx in cand_sorted:
                tag_name = str(names[int(idx)])
                merged[tag_name] = (float(row_arr[int(idx)]), int(cats[int(idx)]))
            merged = self._merge_candidate_copyrights(merged, row_arr, name_to_idx)

            raw_predictions: list[TagPrediction] = []
            for tag_name, (score, category_value) in merged.items():
                # プレースホルダは常に捨てる（ユーザーに出さない）
                if tag_name.startswith(BROKEN_TAG_PREFIX):
                    continue
                try:
                    category = TagCategory(int(category_value))
                except ValueError:
                    continue
                threshold = max(float(resolved_thr.get(category, 0.0)), score_floor)
                if float(score) < threshold:
                    continue
                raw_predictions.append(TagPrediction(name=tag_name, score=float(score), category=category))

            ordered = sorted(raw_predictions, key=lambda pred: (-pred.score, pred.name))
            taken: list[TagPrediction] = []
            per_category: dict[int, int] = {}
            for prediction in ordered:
                if len(taken) >= hard_cap:
                    break
                cat_key = int(prediction.category)
                limit = cat_limits.get(cat_key)
                current = per_category.get(cat_key, 0)
                if limit is not None and current >= limit:
                    continue
                per_category[cat_key] = current + 1
                taken.append(prediction)

            results.append(TagResult(tags=taken))

        return results

    def _merge_candidate_copyrights(
        self,
        candidates: Mapping[str, tuple[float, int | TagCategory]],
        row: np.ndarray,
        name_to_idx: Mapping[str, int],
    ) -> dict[str, tuple[float, int]]:
        """Merge copyright tags derived from candidate character predictions."""

        merged: dict[str, tuple[float, int]] = {
            name: (float(score), int(category)) for name, (score, category) in candidates.items()
        }
        for name, (score, category_value) in list(merged.items()):
            try:
                category = TagCategory(int(category_value))
            except ValueError:
                continue
            if category != TagCategory.CHARACTER:
                continue
            meta = self._tag_meta_index.get(name)
            if not meta or not meta.ips:
                continue
            for ip_name in meta.ips:
                ip_score = float(score)
                ip_idx = name_to_idx.get(ip_name)
                if ip_idx is not None and 0 <= ip_idx < row.shape[0]:
                    ip_score = max(ip_score, float(row[ip_idx]))
                existing = merged.get(ip_name)
                if existing is not None:
                    ip_score = max(ip_score, float(existing[0]))
                merged[ip_name] = (ip_score, int(TagCategory.COPYRIGHT))
        return merged

    def predict(self, image: np.ndarray) -> list[tuple[str, float, TagCategory]]:
        """Run inference over a single preprocessed RGB image array."""

        batch = self.prepare_batch_from_rgb_np([image])
        results = self.infer_batch_prepared(batch)
        if not results:
            return []
        return [(pred.name, pred.score, pred.category) for pred in results[0].tags]


__all__ = ["PixaiOnnxTagger"]
