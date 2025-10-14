"""ONNX Runtime implementation of the PixAI tagger."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from tagger.base import TagCategory, TagPrediction, TagResult, ThresholdMap, MaxTagsMap
from tagger.wd14_onnx import WD14Tagger, _Label

logger = logging.getLogger(__name__)


_CATEGORY_MAP: dict[str, TagCategory] = {
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


class PixaiOnnxTagger(WD14Tagger):
    """WD14-compatible tagger for PixAI ONNX models."""

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
        self._default_input_size = int(input_size)
        super().__init__(
            model_path,
            labels_csv=labels_csv,
            tags_csv=tags_csv,
            providers=providers,
            input_size=input_size,
            default_thresholds=default_thresholds,
            default_max_tags=default_max_tags,
        )
        self._pixai_layout = "NHWC"
        self._pixai_channels_first = False
        self._pixai_channels = 3
        self._determine_input_layout()
        logger.info(
            "PixAI: layout=%s channels=%d size=%d",
            self._pixai_layout,
            self._pixai_channels,
            self._input_size,
        )

    def prepare_batch_from_bgr(self, bgr_list: Sequence[np.ndarray]) -> np.ndarray:
        size = int(self._input_size)
        out = np.empty((len(bgr_list), size, size, 3), dtype=np.float32)
        for i, im in enumerate(bgr_list):
            if im is None:
                out[i].fill(255.0)
                continue
            if im.ndim == 3 and im.shape[2] == 4:
                bgr = im[:, :, :3].astype(np.float32)
                alpha = im[:, :, 3:4].astype(np.float32) / 255.0
                bgr = bgr * alpha + 255.0 * (1.0 - alpha)
                bgr = bgr.astype(np.uint8)
            else:
                bgr = im
            h, w = bgr.shape[:2]
            side = max(h, w, size)
            top = (side - h) // 2
            left = (side - w) // 2
            sq = cv2.copyMakeBorder(
                bgr,
                top,
                side - h - top,
                left,
                side - w - left,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            interp = cv2.INTER_CUBIC if side < size else cv2.INTER_AREA
            resized = cv2.resize(sq, (size, size), interpolation=interp)
            out[i] = resized.astype(np.float32)
        return out

    def prepare_batch_pil(self, images: list[Image.Image]) -> np.ndarray:
        size = int(self._input_size)
        batch = np.empty((len(images), size, size, 3), dtype=np.float32)
        for i, image in enumerate(images):
            batch[i] = self._preprocess_np(image)
        return batch

    def prepare_batch_from_rgb_np(self, imgs_rgb: Sequence[np.ndarray]) -> np.ndarray:
        size = int(self._input_size)
        batch = np.empty((len(imgs_rgb), size, size, 3), dtype=np.float32)
        for i, arr in enumerate(imgs_rgb):
            if arr is None:
                batch[i].fill(255.0)
                continue
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 4:
                rgb = arr[:, :, :3].astype(np.float32)
                alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
                rgb = rgb * alpha + 255.0 * (1.0 - alpha)
                arr = rgb.astype(np.uint8)
            bgr = arr[:, :, ::-1]
            bgr = self.make_square(bgr, size)
            bgr = self.smart_resize(bgr, size)
            batch[i] = bgr.astype(np.float32, copy=False)
        return batch

    def _run_session(self, batch: np.ndarray) -> Sequence[np.ndarray]:
        feed = batch
        if self._pixai_channels_first:
            feed = np.transpose(feed, (0, 3, 1, 2))
        feed = np.ascontiguousarray(feed, dtype=np.float32)
        return self._session.run(self._output_names, {self._input_name: feed})

    def _augment_result(
        self,
        result: TagResult,
        indices: Sequence[int],
        scores: Sequence[float],
    ) -> None:
        valid_mask = getattr(self, "_valid_label_mask", None)
        extras: list[TagPrediction] = []
        existing = {tag.name for tag in result.tags}
        for index, score in zip(indices, scores):
            if index >= len(self._label_ips):
                continue
            if valid_mask is not None and not valid_mask[index]:
                continue
            label = self._labels[index]
            if TagCategory(int(label.category)) != TagCategory.CHARACTER:
                continue
            for name in label.ips:
                tag_name = str(name).strip()
                if not tag_name or tag_name in existing:
                    continue
                extras.append(
                    TagPrediction(
                        name=tag_name,
                        score=float(score),
                        category=TagCategory.COPYRIGHT,
                    )
                )
                existing.add(tag_name)
        if extras:
            result.tags.extend(extras)

    def _determine_input_layout(self) -> None:
        input_info = self._session.get_inputs()[0]
        shape = getattr(input_info, "shape", [])
        dims = [self._normalise_dim(dim) for dim in shape]
        default = self._default_input_size
        layout = "NHWC"
        height = default
        width = default
        channels = 3
        if len(dims) >= 4:
            n, c, h, w = dims[:4]
            if 0 < c <= 4 and (len(dims) == 4 and (dims[-1] > 4 or dims[-1] <= 0)):
                layout = "NCHW"
                channels = c if c > 0 else channels
                height = h if h > 0 else default
                width = w if w > 0 else height
            else:
                layout = "NHWC"
                height = dims[1] if dims[1] > 0 else (dims[2] if len(dims) > 2 and dims[2] > 0 else default)
                width = dims[2] if len(dims) > 2 and dims[2] > 0 else height
                channels = dims[3] if len(dims) > 3 and dims[3] > 0 else channels
        elif len(dims) >= 3:
            layout = "NHWC"
            height = dims[1] if dims[1] > 0 else default
            width = dims[2] if dims[2] > 0 else height
        self._pixai_layout = layout
        self._pixai_channels_first = layout == "NCHW"
        self._pixai_channels = channels
        resolved_size = height if height > 0 else default
        if resolved_size <= 0:
            resolved_size = default
        self._input_size = resolved_size

    @staticmethod
    def _normalise_dim(value: object) -> int:
        if value is None:
            return -1
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"none", "nan"}:
                return -1
            try:
                return int(float(stripped))
            except ValueError:
                return -1
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _parse_category(value: str | None) -> TagCategory:
        if not value:
            return TagCategory.GENERAL
        lowered = value.strip().lower()
        if lowered in _CATEGORY_MAP:
            return _CATEGORY_MAP[lowered]
        try:
            return TagCategory(int(float(lowered)))
        except (TypeError, ValueError):
            return TagCategory.GENERAL

    @staticmethod
    def _parse_count(value: str | None) -> int:
        if not value:
            return 0
        try:
            return int(float(value.strip()))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _parse_ips(value: str | None) -> tuple[str, ...]:
        if not value:
            return ()
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("PixAI: failed to parse ips field '%s'", text)
            return ()
        if isinstance(parsed, str):
            items = [parsed]
        elif isinstance(parsed, (list, tuple)):
            items = list(parsed)
        else:
            return ()
        cleaned: list[str] = []
        for item in items:
            name = str(item).strip()
            if name:
                cleaned.append(name)
        # preserve order while removing duplicates
        seen: dict[str, None] = {}
        for name in cleaned:
            seen.setdefault(name, None)
        return tuple(seen.keys())

    @staticmethod
    def _load_labels(labels_csv: str | Path) -> list[_Label]:
        path = Path(labels_csv)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if not rows:
            raise ValueError("PixAI: label CSV is empty")
        header = [cell.strip() for cell in rows[0]]
        lower_header = [cell.lower() for cell in header]
        has_header = "name" in lower_header
        field_map = {name.lower(): idx for idx, name in enumerate(header)} if has_header else {}
        name_idx = field_map.get("name") if has_header else (2 if len(header) > 2 else None)
        category_idx = field_map.get("category") if has_header else (3 if len(header) > 3 else None)
        count_idx = field_map.get("count") if has_header else (4 if len(header) > 4 else None)
        ips_idx = field_map.get("ips") if has_header else (5 if len(header) > 5 else None)
        labels: list[_Label] = []
        data_rows = rows[1:] if has_header else rows
        for raw in data_rows:
            if not raw:
                labels.append(_Label(name="", category=TagCategory.GENERAL, count=0, ips=()))
                continue
            cells = [cell.strip() for cell in raw]
            name = cells[name_idx] if name_idx is not None and name_idx < len(cells) else ""
            category_value = cells[category_idx] if category_idx is not None and category_idx < len(cells) else "0"
            count_value = cells[count_idx] if count_idx is not None and count_idx < len(cells) else "0"
            ips_value = cells[ips_idx] if ips_idx is not None and ips_idx < len(cells) else "[]"
            category = PixaiOnnxTagger._parse_category(category_value)
            count = PixaiOnnxTagger._parse_count(count_value)
            ips = PixaiOnnxTagger._parse_ips(ips_value)
            labels.append(_Label(name=name, category=category, count=count, ips=ips))
        if not labels:
            raise ValueError("PixAI: no labels parsed from CSV")
        return labels


__all__ = ["PixaiOnnxTagger"]
