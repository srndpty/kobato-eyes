"""ONNX Runtime tagger for the PixAI label set with copyright enrichment."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

from tagger.base import MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap
from tagger.labels_util import BROKEN_TAG_PREFIX, TagMeta, load_selected_tags

from .wd14_onnx import WD14Tagger, _sigmoid

logger = logging.getLogger(__name__)


def _normalize_np_chw(x: np.ndarray, mean, std):
    # x: (3,H,W) in [0,1]
    x = x.astype(np.float32, copy=False)
    for c in range(3):
        x[c] = (x[c] - mean[c]) / std[c]
    return x


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

        raise RuntimeError("PixAI: unable to determine prediction tensor from outputs " f"{output_names}")

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
        except Exception:
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

        # __init__ の末尾で呼ぶ
        self._try_fix_label_order_with_json()

    def _try_fix_label_order_with_json(self):
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
        expected = [None] * N
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
        self._label_name_cache = expected

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

        # 分布を再ログ
        vals, cnts = np.unique(np.array(self._label_cat_cache, dtype=np.int32), return_counts=True)
        logger.info("PixAI labels category distribution (after fix): %s", {int(v): int(c) for v, c in zip(vals, cnts)})

    def prepare_batch_from_rgb_np(self, rgb_list: list[np.ndarray]) -> np.ndarray:
        assert len(rgb_list) >= 1, "rgb_list empty"
        # 入力検証
        a0 = rgb_list[0]
        logger.info(
            "PixAI prepare: in[0] shape=%s dtype=%s min=%.3f max=%.3f",
            getattr(a0, "shape", None),
            getattr(a0, "dtype", None),
            float(a0.min()),
            float(a0.max()),
        )
        out = []
        # preprocess.json があれば mean/std を使い、なければ fallback を使う
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        target = int(self._input_size)  # 例: 448

        # preprocess.json から mean/std を拾えるよう軽いパーサ（なければデフォルト0.5/0.5）
        for st in self._pixai_pp or []:
            t = str(st.get("type") or st.get("op") or "").lower()
            if t == "normalize":
                mean = st.get("mean", mean)
                std = st.get("std", std)

        for arr in rgb_list:
            # 1) PIL にして処理（アルファは白合成済みの想定。なければここで合成）
            im = Image.fromarray(arr, mode="RGB")

            # 2) PixAI はだいたい「短辺基準リサイズ + 中央クロップ」が多い
            #    preprocess.json を厳密に再現できるならそれに従うのが最善。
            #    ここでは汎用的な fallback を実装：
            w, h = im.size
            scale = target / min(w, h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            if (nw, nh) != (w, h):
                im = im.resize((nw, nh), Image.BICUBIC)

            # 中央クロップで target x target
            left = (im.width - target) // 2
            top = (im.height - target) // 2
            im = im.crop((left, top, left + target, top + target))

            # 3) [H,W,3] -> [3,H,W], [0..255] -> [0..1] -> normalize
            x = np.asarray(im, dtype=np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            x = _normalize_np_chw(x, mean, std)

            out.append(x)

        # NCHW float32
        batch = np.stack(out, axis=0)
        logger.info(
            "PixAI prepare: out batch=%s dtype=%s min=%.3f max=%.3f",
            batch.shape,
            batch.dtype,
            float(batch.min()),
            float(batch.max()),
        )
        return batch

    @staticmethod
    def _build_tag_meta_index(labels_csv: Path) -> dict[str, TagMeta]:
        index: dict[str, TagMeta] = {}
        try:
            for meta in load_selected_tags(labels_csv):
                index[meta.name] = meta
        except Exception:
            logger.warning("PixAI: failed to parse tag metadata from %s", labels_csv)
        return index

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

        cat_thresholds: dict[int, float] = {}
        for category in TagCategory:
            base_thr = float(resolved_thr.get(category, 0.0))
            cat_thresholds[int(category)] = max(base_thr, score_floor)

        cat_limits: dict[int, int | None] = {}
        for category in TagCategory:
            limit_value = resolved_limits.get(category)
            cat_limits[int(category)] = int(limit_value) if limit_value is not None else None

        names = self._label_name_cache
        # cats = self._label_cat_cache
        cats = self._effective_cats
        if logits.shape[1] != len(names):
            raise RuntimeError(f"Model output dim {logits.shape[1]} != labels {len(names)}")

        results: list[TagResult] = []
        hard_cap = max(1, int(getattr(self, "_topk_cap", 128)))

        for row in probs:
            # 先頭1枚だけ実インデックスと現在マッピング名をダンプ
            if not hasattr(self, "_dbg_top_once"):
                self._dbg_top_once = True
                row = probs[0]  # 先頭画像
                top_idx = np.argsort(row)[-10:][::-1]
                pairs = [(int(i), self._label_name_cache[int(i)], float(row[int(i)])) for i in top_idx]
                logger.info("PixAI TOP10 indices (current mapping): %s", pairs)
            base = {names[idx]: (float(row[idx]), cats[idx]) for idx in range(len(names))}
            merged = self._merge_copyrights(base)
            raw_predictions: list[TagPrediction] = []
            for tag_name, (score, category_value) in merged.items():
                # プレースホルダは常に捨てる（ユーザーに出さない）
                if tag_name.startswith(BROKEN_TAG_PREFIX):
                    continue
                try:
                    category = TagCategory(int(category_value))
                except ValueError:
                    continue
                threshold = cat_thresholds.get(int(category), max(float(resolved_thr.get(category, 0.0)), score_floor))
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

    def predict(self, image: np.ndarray) -> list[tuple[str, float, TagCategory]]:
        """Run inference over a single preprocessed RGB image array."""

        batch = self.prepare_batch_from_rgb_np([image])
        results = self.infer_batch_prepared(batch)
        if not results:
            return []
        return [(pred.name, pred.score, pred.category) for pred in results[0].tags]


__all__ = ["PixaiOnnxTagger"]
