"""PyTorch implementation of the PixAI Danbooru tagger."""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from tagger.base import (
    ITagger,
    MaxTagsMap,
    TagCategory,
    TagPrediction,
    TagResult,
    ThresholdMap,
)

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency guard
    import timm
except Exception as exc:  # pragma: no cover - runtime guard
    timm = None  # type: ignore[assignment]
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

_MODEL_FILENAME = "model_v0.9.pth"
_TAGS_FILENAME = "tags_v0.9_13k.json"
_CHAR_IP_FILENAME = "char_ip_map.json"
_ENCODER_NAME = "wd-eva02-large"
_FEATURE_DIM = 1024

_DEFAULT_THRESHOLDS = {
    TagCategory.GENERAL: 0.30,
    TagCategory.CHARACTER: 0.85,
    TagCategory.COPYRIGHT: 0.30,
}


@dataclass(frozen=True)
class _TagInfo:
    index: int
    name: str
    category: TagCategory


def _ensure_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "PixAI tagger requires torch. Install optional dependency group 'pixai' to enable it."
        ) from _TORCH_IMPORT_ERROR


def _ensure_timm() -> None:
    if timm is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "PixAI tagger requires timm. Install optional dependency group 'pixai' to enable it."
        ) from _TIMM_IMPORT_ERROR


def _coerce_category(value: Any) -> TagCategory:
    if isinstance(value, TagCategory):
        return value
    if value is None:
        return TagCategory.GENERAL
    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {
            "general": TagCategory.GENERAL,
            "tag": TagCategory.GENERAL,
            "character": TagCategory.CHARACTER,
            "char": TagCategory.CHARACTER,
            "copyright": TagCategory.COPYRIGHT,
            "ip": TagCategory.COPYRIGHT,
            "rating": TagCategory.RATING,
            "artist": TagCategory.ARTIST,
            "meta": TagCategory.META,
        }
        return mapping.get(lowered, TagCategory.GENERAL)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return TagCategory.GENERAL
    mapping = {
        0: TagCategory.GENERAL,
        1: TagCategory.ARTIST,
        2: TagCategory.COPYRIGHT,
        3: TagCategory.CHARACTER,
        4: TagCategory.META,
        5: TagCategory.RATING,
    }
    return mapping.get(numeric, TagCategory.GENERAL)


def _normalise_tag_entries(raw: Any) -> list[_TagInfo]:
    entries: list[_TagInfo] = []
    if isinstance(raw, dict):
        if "tags" in raw and isinstance(raw["tags"], list):
            iterable: Iterable[Any] = raw["tags"]
        elif all(isinstance(v, list) for v in raw.values()):
            expanded: list[dict[str, Any]] = []
            for key, values in raw.items():
                for value in values:
                    if isinstance(value, dict):
                        item = dict(value)
                    else:
                        item = {"name": value}
                    item.setdefault("category", key)
                    expanded.append(item)
            iterable = expanded
        else:
            iterable = [{"name": key, "index": idx, "category": value} for idx, (key, value) in enumerate(raw.items())]
    elif isinstance(raw, list):
        iterable = raw
    else:
        iterable = []

    next_index = 0
    for item in iterable:
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            category = TagCategory.GENERAL
            index = next_index
        elif isinstance(item, Mapping):
            name_source = item.get("name") or item.get("tag") or item.get("label") or item.get("value")
            if not name_source:
                continue
            name = str(name_source)
            category = _coerce_category(item.get("category") or item.get("type"))
            index_value = item.get("index")
            if index_value is None:
                index_value = item.get("id") or item.get("class_index")
            if index_value is None:
                index = next_index
            else:
                try:
                    index = int(index_value)
                except (TypeError, ValueError):
                    index = next_index
        else:
            continue
        entries.append(_TagInfo(index=index, name=name, category=category))
        next_index = max(next_index, index + 1)

    entries.sort(key=lambda info: info.index)
    # Re-assign to dense indices to match classifier expectations.
    dense_entries = [
        _TagInfo(index=idx, name=info.name, category=info.category) for idx, info in enumerate(entries)
    ]
    return dense_entries


class _PixAIModel(nn.Module):  # type: ignore[misc]
    """Composite PixAI tagger model."""

    def __init__(self, encoder: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(FEATURE_DIM, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised indirectly
        features = self.encoder.forward_features(inputs)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.head(features)
        return self.activation(logits)


class PixAITagger(ITagger):
    """PixAI tagger backed by PyTorch and timm."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device | None = None,
        default_thresholds: ThresholdMap | None = None,
    ) -> None:
        _ensure_torch()
        _ensure_timm()
        assert torch is not None  # for type-checkers
        assert nn is not None

        self._model_dir = Path(model_dir)
        if not self._model_dir.exists():
            raise FileNotFoundError(f"PixAI model directory not found: {self._model_dir}")
        if not self._model_dir.is_dir():
            raise NotADirectoryError(f"PixAI model directory is not a directory: {self._model_dir}")

        self._tags = self._load_tags(self._model_dir / _TAGS_FILENAME)
        if not self._tags:
            raise RuntimeError("PixAI: tags file contained no tag definitions")
        self._char_to_ip = self._load_char_ip_map(self._model_dir / _CHAR_IP_FILENAME)

        self._default_thresholds = {
            **_DEFAULT_THRESHOLDS,
            **self._normalise_thresholds(default_thresholds or {}),
        }

        resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = resolved_device
        logger.info("PixAI: using device %s", self._device)

        num_classes = len(self._tags)
        encoder = timm.create_model(_ENCODER_NAME, pretrained=False)
        if hasattr(encoder, "reset_classifier"):
            encoder.reset_classifier(0)
        self._model = _PixAIModel(encoder, num_classes)
        state_path = self._model_dir / _MODEL_FILENAME
        self._load_state_dict(state_path)
        self._model = self._model.to(self._device)
        self._model.eval()

        self._autocast_enabled = self._device.type == "cuda"

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def prepare_batch_from_rgb_np(self, imgs_rgb: Sequence[np.ndarray]) -> np.ndarray:
        if not imgs_rgb:
            return np.empty((0, 448, 448, 3), dtype=np.float32)
        batch = np.stack([np.asarray(img, dtype=np.float32, order="C") for img in imgs_rgb], axis=0)
        return batch

    def infer_batch_prepared(
        self,
        np_batch: np.ndarray,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        if np_batch.ndim != 4 or np_batch.shape[-1] != 3:
            raise ValueError("PixAI: expected (B,H,W,3) batch for inference")
        if torch is None:
            raise RuntimeError("PixAI tagger unavailable (torch not imported)")

        tensor = torch.from_numpy(np_batch).float()
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.div(255.0).mul(2.0).sub(1.0)
        if self._device.type == "cuda":
            tensor = tensor.pin_memory()
        tensor = tensor.to(self._device, non_blocking=True)

        with torch.inference_mode():
            if self._autocast_enabled:
                context = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                context = nullcontext()
            with context:
                probs = self._model(tensor)
        probs = probs.detach().to("cpu", non_blocking=True).float().numpy()

        resolved_thresholds = self._resolve_thresholds(thresholds)
        resolved_max_tags = self._resolve_max_tags(max_tags)

        results: list[TagResult] = []
        for row in probs:
            predictions_by_cat: dict[TagCategory, list[TagPrediction]] = {
                TagCategory.GENERAL: [],
                TagCategory.CHARACTER: [],
                TagCategory.COPYRIGHT: [],
            }
            character_scores: dict[str, float] = {}
            for info, score in zip(self._tags, row):
                category = info.category
                threshold = resolved_thresholds.get(category, 0.0)
                if score < threshold:
                    continue
                prediction = TagPrediction(name=info.name, score=float(score), category=category)
                predictions_by_cat.setdefault(category, []).append(prediction)
                if category == TagCategory.CHARACTER:
                    character_scores[info.name] = max(character_scores.get(info.name, 0.0), float(score))

            copyright_scores: dict[str, float] = {}
            for char_name, score in character_scores.items():
                for ip in self._char_to_ip.get(char_name, ()):  # type: ignore[arg-type]
                    copyright_scores[ip] = max(copyright_scores.get(ip, 0.0), score)
            copyright_threshold = resolved_thresholds.get(TagCategory.COPYRIGHT, resolved_thresholds.get(TagCategory.GENERAL, 0.0))
            for ip_name, score in copyright_scores.items():
                if score < copyright_threshold:
                    continue
                predictions_by_cat[TagCategory.COPYRIGHT].append(
                    TagPrediction(name=ip_name, score=score, category=TagCategory.COPYRIGHT)
                )

            all_predictions: list[TagPrediction] = []
            for category in (TagCategory.GENERAL, TagCategory.CHARACTER, TagCategory.COPYRIGHT):
                preds = predictions_by_cat.get(category, [])
                if not preds:
                    continue
                preds.sort(key=lambda pred: (-pred.score, pred.name))
                limit = resolved_max_tags.get(category)
                if limit is not None and limit > 0:
                    preds = preds[:limit]
                all_predictions.extend(preds)
            results.append(TagResult(tags=all_predictions))
        return results

    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds: ThresholdMap | None = None,
        max_tags: MaxTagsMap | None = None,
    ) -> list[TagResult]:
        rgb_arrays = [np.array(image.convert("RGB"), dtype=np.float32) for image in images]
        batch = self.prepare_batch_from_rgb_np(rgb_arrays)
        return self.infer_batch_prepared(batch, thresholds=thresholds, max_tags=max_tags)

    def _load_state_dict(self, state_path: Path) -> None:
        assert torch is not None
        if not state_path.exists():
            raise FileNotFoundError(f"PixAI model weights not found: {state_path}")
        logger.info("PixAI: loading weights from %s", state_path)
        state = torch.load(state_path, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], Mapping):
                state_dict = state["state_dict"]
            elif "model" in state and isinstance(state["model"], Mapping) and "state_dict" in state["model"]:
                state_dict = state["model"]["state_dict"]
            else:
                state_dict = state
        else:
            raise RuntimeError("PixAI: unexpected checkpoint format")
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("PixAI: missing parameters during load: %s", ", ".join(sorted(missing)))
        if unexpected:
            logger.debug("PixAI: unexpected parameters during load: %s", ", ".join(sorted(unexpected)))

    def _load_tags(self, tags_path: Path) -> list[_TagInfo]:
        if not tags_path.exists():
            raise FileNotFoundError(f"PixAI tags file not found: {tags_path}")
        logger.info("PixAI: loading tags from %s", tags_path)
        data = json.loads(tags_path.read_text(encoding="utf-8"))
        return _normalise_tag_entries(data)

    def _load_char_ip_map(self, mapping_path: Path) -> dict[str, list[str]]:
        if not mapping_path.exists():
            logger.warning("PixAI: char→IP mapping file missing (%s); copyright tags disabled", mapping_path)
            return {}
        logger.info("PixAI: loading character IP map from %s", mapping_path)
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping: dict[str, list[str]] = {}
        if isinstance(data, Mapping):
            for key, value in data.items():
                if not isinstance(key, str):
                    continue
                if isinstance(value, str):
                    mapping[key] = [value]
                elif isinstance(value, Iterable):
                    mapping[key] = [str(item) for item in value if isinstance(item, (str, int))]
        return mapping

    def _resolve_thresholds(self, overrides: ThresholdMap | None) -> dict[TagCategory, float]:
        resolved = dict(self._default_thresholds)
        if not overrides:
            return resolved
        for key, value in overrides.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            category = self._coerce_category_key(key)
            resolved[category] = numeric
        return resolved

    def _resolve_max_tags(self, overrides: MaxTagsMap | None) -> dict[TagCategory, int | None]:
        resolved: dict[TagCategory, int | None] = {}
        if not overrides:
            return resolved
        for key, value in overrides.items():
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            category = self._coerce_category_key(key)
            resolved[category] = numeric if numeric > 0 else None
        return resolved

    @staticmethod
    def _coerce_category_key(key: Any) -> TagCategory:
        if isinstance(key, TagCategory):
            return key
        if isinstance(key, str):
            lowered = key.strip().lower()
            mapping = {
                "general": TagCategory.GENERAL,
                "character": TagCategory.CHARACTER,
                "copyright": TagCategory.COPYRIGHT,
                "artist": TagCategory.ARTIST,
                "rating": TagCategory.RATING,
                "meta": TagCategory.META,
            }
            if lowered in mapping:
                return mapping[lowered]
        return TagCategory.GENERAL

    def _normalise_thresholds(self, thresholds: ThresholdMap) -> dict[TagCategory, float]:
        resolved: dict[TagCategory, float] = {}
        for key, value in thresholds.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            resolved[self._coerce_category_key(key)] = numeric
        return resolved


__all__ = ["PixAITagger"]
