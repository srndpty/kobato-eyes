"""ONNX Runtime implementation for the PixAI tagger."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from tagger.base import MaxTagsMap, TagCategory, TagPrediction, TagResult, ThresholdMap

from .wd14_onnx import WD14Tagger, _Label

logger = logging.getLogger(__name__)


class PixaiOnnxTagger(WD14Tagger):
    """PixAI tagger compatible with the WD14 interface."""

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
        self._pixai_valid_mask: np.ndarray | None = None
        self._pixai_ips: list[tuple[str, ...]] = []
        self._input_layout = "NHWC"
        self._input_height = int(input_size)
        self._input_width = int(input_size)
        super().__init__(
            model_path,
            labels_csv=labels_csv,
            tags_csv=tags_csv,
            providers=providers,
            input_size=input_size,
            default_thresholds=default_thresholds,
            default_max_tags=default_max_tags,
        )
        if self._pixai_valid_mask is None or self._pixai_valid_mask.size != len(self._labels):  # type: ignore[attr-defined]
            self._pixai_valid_mask = np.ones(len(self._labels), dtype=bool)  # type: ignore[attr-defined]
        else:
            self._pixai_valid_mask = self._pixai_valid_mask.astype(bool, copy=False)
        # Mask invalid rows in the cached threshold vector
        self._default_thr_vec = np.where(self._pixai_valid_mask, self._default_thr_vec, np.inf)
        self._configure_input_geometry()

    def _configure_input_geometry(self) -> None:
        """Determine input layout and spatial dimensions from the ONNX graph."""

        input_meta = self._session.get_inputs()[0]  # type: ignore[attr-defined]
        raw_shape = list(getattr(input_meta, "shape", []))
        dims: list[int | None] = []
        for dim in raw_shape:
            if isinstance(dim, (int, np.integer)) and int(dim) > 0:
                dims.append(int(dim))
            elif isinstance(dim, str):
                try:
                    value = int(dim)
                except ValueError:
                    dims.append(None)
                else:
                    dims.append(value if value > 0 else None)
            else:
                dims.append(None)

        layout = "NHWC"
        height: int | None = None
        width: int | None = None
        channel_axis: int | None = None
        for idx, dim in enumerate(dims):
            if dim in (1, 3):
                channel_axis = idx
                break

        if len(dims) == 4 and channel_axis is not None:
            if channel_axis == 1:  # NCHW
                layout = "NCHW"
                height = dims[2]
                width = dims[3]
            elif channel_axis == 3:  # NHWC
                layout = "NHWC"
                height = dims[1]
                width = dims[2]
        if height is None and len(dims) >= 3:
            height = dims[-2]
        if width is None and len(dims) >= 2:
            width = dims[-1]

        fallback = int(getattr(self, "_input_size", 448))
        if not height or height <= 0:
            height = fallback
        if not width or width <= 0:
            width = height

        self._input_layout = layout
        self._input_height = int(height)
        self._input_width = int(width)
        self._input_size = int(height)
        logger.info(
            "PixAI: input layout=%s height=%d width=%d", self._input_layout, self._input_height, self._input_width
        )

    def _load_labels(self, labels_csv: str | Path) -> list[_Label]:  # type: ignore[override]
        path = Path(labels_csv)
        labels: list[_Label] = []
        valid_mask: list[bool] = []
        ips_entries: list[tuple[str, ...]] = []

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                cells = [cell.strip() for cell in row]
                if not cells:
                    continue
                lower = cells[0].lower()
                if lower.startswith("#"):
                    continue
                if lower in {"id", "tag_id"}:
                    continue
                padded = (cells + [""] * 6)[:6]
                name = padded[2].strip()
                category_value = padded[3]
                count_value = padded[4]
                ips_value = padded[5]

                try:
                    category = TagCategory(int(category_value))
                except Exception:
                    category = TagCategory.GENERAL
                try:
                    count = int(float(count_value))
                except Exception:
                    count = 0

                if name:
                    labels.append(_Label(name=name, category=category, count=count))
                    valid_mask.append(True)
                else:
                    labels.append(_Label(name="", category=category, count=count))
                    valid_mask.append(False)

                try:
                    ips_list = json.loads(ips_value)
                except Exception:
                    ips_list = []
                if not isinstance(ips_list, list):
                    ips_list = []
                cleaned_ips = tuple(str(item).strip() for item in ips_list if str(item).strip())
                ips_entries.append(cleaned_ips)

        if not labels:
            raise ValueError("PixAI: no labels parsed from CSV")

        self._pixai_valid_mask = np.array(valid_mask, dtype=bool)
        self._pixai_ips = ips_entries
        return labels

    def _build_threshold_vector(self, thresholds: dict[TagCategory | int, float]) -> np.ndarray:  # type: ignore[override]
        vec = super()._build_threshold_vector(thresholds)
        if self._pixai_valid_mask is not None and self._pixai_valid_mask.size == vec.size:
            vec = vec.astype(np.float32, copy=True)
            vec[~self._pixai_valid_mask] = np.inf
        return vec

    def prepare_batch_from_rgb_np(self, imgs_rgb: Sequence[np.ndarray]) -> np.ndarray:  # type: ignore[override]
        target = int(self._input_height)
        batch: list[np.ndarray] = []
        for arr in imgs_rgb:
            if arr is None:
                canvas = np.full((target, target, 3), 255.0, dtype=np.float32)
                batch.append(canvas)
                continue
            work = arr
            if work.ndim == 2:
                work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)
            elif work.shape[2] == 4:
                rgb = work[:, :, :3].astype(np.float32)
                alpha = work[:, :, 3:4].astype(np.float32) / 255.0
                rgb = rgb * alpha + 255.0 * (1.0 - alpha)
                work = rgb.astype(np.uint8)
            bgr = work[:, :, ::-1]
            bgr = self.make_square(bgr, target)
            bgr = self.smart_resize(bgr, target)
            batch.append(bgr.astype(np.float32))

        stacked = np.stack(batch, axis=0) if batch else np.empty((0, target, target, 3), dtype=np.float32)
        return self._finalise_batch(stacked)

    def prepare_batch_from_bgr(self, bgr_list: list[np.ndarray]) -> np.ndarray:  # type: ignore[override]
        target = int(self._input_height)
        batch: list[np.ndarray] = []
        for image in bgr_list:
            if image is None:
                canvas = np.full((target, target, 3), 255.0, dtype=np.float32)
                batch.append(canvas)
                continue
            work = image
            if work.ndim == 2:
                work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
            elif work.shape[2] == 4:
                bgr = work[:, :, :3].astype(np.float32)
                alpha = work[:, :, 3:4].astype(np.float32) / 255.0
                work = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
            bgr = self.make_square(work, target)
            bgr = self.smart_resize(bgr, target)
            batch.append(bgr.astype(np.float32))

        stacked = np.stack(batch, axis=0) if batch else np.empty((0, target, target, 3), dtype=np.float32)
        return self._finalise_batch(stacked)

    def prepare_batch_pil(self, images: list[Image.Image]) -> np.ndarray:  # type: ignore[override]
        batch = [self._preprocess_np(image) for image in images]
        stacked = np.stack(batch, axis=0) if batch else np.empty((0, self._input_height, self._input_width, 3), dtype=np.float32)
        return self._finalise_batch(stacked)

    def _preprocess_np(self, image: Image.Image) -> np.ndarray:  # type: ignore[override]
        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, "WHITE")
        canvas.paste(image, mask=image)
        rgb = np.asarray(canvas.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        bgr = self.make_square(bgr, self._input_height)
        bgr = self.smart_resize(bgr, self._input_height)
        return bgr.astype(np.float32)

    def _finalise_batch(self, batch: np.ndarray) -> np.ndarray:
        if self._input_layout == "NHWC":
            return batch
        return np.transpose(batch, (0, 3, 1, 2))

    def _postprocess_logits_topk(  # type: ignore[override]
        self,
        logits: np.ndarray,
        *,
        thresholds: ThresholdMap | None,
        max_tags: MaxTagsMap | None,
    ) -> list[TagResult]:
        if logits.dtype != np.float32:
            logits = logits.astype(np.float32, copy=False)
        mn, mx = float(np.min(logits)), float(np.max(logits))
        probs = logits if (0.0 <= mn <= 1.0 and 0.0 <= mx <= 1.0) else 1.0 / (1.0 + np.exp(-logits))

        B, C = probs.shape
        valid_mask = self._pixai_valid_mask
        if valid_mask is None or valid_mask.size != C:
            valid_mask = np.ones(C, dtype=bool)

        resolved_thr = self._resolve_thresholds(self._default_thresholds, thresholds)
        thr_vec = self._build_threshold_vector(resolved_thr)

        resolved_limits = self._resolve_max_tags(self._default_max_tags, max_tags)
        has_unbounded = any(v is None for v in resolved_limits.values())
        base_cap = None if has_unbounded else max(sum(int(v) for v in resolved_limits.values() if v is not None), 64)
        hard_cap = max(1, int(self._topk_cap))

        names = self._label_names
        cats = self._label_cats
        results: list[TagResult] = []

        for b in range(B):
            scores = probs[b]
            hit_mask = (scores >= thr_vec) & valid_mask
            hit_count = int(hit_mask.sum())
            if hit_count == 0:
                results.append(TagResult(tags=[]))
                continue

            if base_cap is None:
                K = min(hit_count, hard_cap)
            else:
                K = min(hit_count, base_cap, hard_cap)

            masked = np.where(hit_mask, scores, -np.inf)
            kth = max(0, min(K - 1, C - 1))
            cand_idx = np.argpartition(-masked, kth)[:K]
            cand_sorted = cand_idx[np.argsort(-masked[cand_idx], kind="stable")]

            base_entries: list[tuple[int, float]] = []
            remaining_for_base = np.full(8, np.iinfo(np.int32).max, dtype=np.int32)
            for cat, lim in resolved_limits.items():
                if lim is not None:
                    idx = int(cat)
                    if 0 <= idx < remaining_for_base.size:
                        remaining_for_base[idx] = max(0, int(lim))

            extra_copyright: dict[str, float] = {}
            for j in cand_sorted:
                if not hit_mask[j]:
                    continue
                category_index = int(cats[j])
                score = float(scores[j])
                if not np.isfinite(score):
                    continue
                if category_index < remaining_for_base.size and remaining_for_base[category_index] <= 0:
                    continue
                base_entries.append((int(j), score))
                if category_index == int(TagCategory.CHARACTER) and 0 <= j < len(self._pixai_ips):
                    for extra in self._pixai_ips[j]:
                        if extra:
                            stored = extra_copyright.get(extra, 0.0)
                            if score > stored:
                                extra_copyright[extra] = score
                if category_index < remaining_for_base.size:
                    remaining_for_base[category_index] -= 1

            base_predictions = [
                TagPrediction(
                    name=str(names[idx]),
                    score=score,
                    category=TagCategory(int(cats[idx])),
                )
                for idx, score in base_entries
                if str(names[idx]).strip()
            ]

            extra_predictions = [
                TagPrediction(name=name, score=score, category=TagCategory.COPYRIGHT)
                for name, score in extra_copyright.items()
                if name
            ]

            combined = base_predictions + extra_predictions
            if not combined:
                results.append(TagResult(tags=[]))
                continue

            combined.sort(key=lambda pred: (-pred.score, pred.name.lower()))

            remaining = np.full(8, np.iinfo(np.int32).max, dtype=np.int32)
            for cat, lim in resolved_limits.items():
                if lim is not None:
                    idx = int(cat)
                    if 0 <= idx < remaining.size:
                        remaining[idx] = max(0, int(lim))

            seen: dict[int, set[str]] = {}
            final_preds: list[TagPrediction] = []
            for pred in combined:
                cat_idx = int(pred.category)
                if cat_idx < remaining.size and remaining[cat_idx] <= 0:
                    continue
                bucket = seen.setdefault(cat_idx, set())
                if pred.name in bucket:
                    continue
                final_preds.append(pred)
                bucket.add(pred.name)
                if cat_idx < remaining.size:
                    remaining[cat_idx] -= 1

            results.append(TagResult(tags=final_preds))

        return results


__all__ = ["PixaiOnnxTagger"]

