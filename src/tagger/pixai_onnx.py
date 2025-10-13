"""ONNX Runtime tagger for the PixAI label set with copyright enrichment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from tagger.base import MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap
from tagger.labels_util import TagMeta, load_selected_tags

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
                    logger.info(
                        "PixAI: selecting output '%s' from available tensors %s", name, output_names
                    )
                return [name]

        if len(output_names) == 1:
            return output_names

        raise RuntimeError(
            "PixAI: unable to determine prediction tensor from outputs " f"{output_names}"
        )

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
            name: (float(score), int(category))
            for name, (score, category) in probs_by_name.items()
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
        cats = self._label_cat_cache
        if logits.shape[1] != len(names):
            raise RuntimeError(f"Model output dim {logits.shape[1]} != labels {len(names)}")

        results: list[TagResult] = []
        hard_cap = max(1, int(getattr(self, "_topk_cap", 128)))

        for row in probs:
            base = {names[idx]: (float(row[idx]), cats[idx]) for idx in range(len(names))}
            merged = self._merge_copyrights(base)
            raw_predictions: list[TagPrediction] = []
            for tag_name, (score, category_value) in merged.items():
                try:
                    category = TagCategory(int(category_value))
                except ValueError:
                    continue
                threshold = cat_thresholds.get(int(category), max(float(resolved_thr.get(category, 0.0)), score_floor))
                if float(score) < threshold:
                    continue
                raw_predictions.append(
                    TagPrediction(name=tag_name, score=float(score), category=category)
                )

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
