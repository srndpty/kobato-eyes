"""Embedding utilities built on top of OpenCLIP."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

from utils.image_io import safe_load_image

logger = logging.getLogger(__name__)


class OpenClipEmbedder:
    """Generate L2-normalized image embeddings using OpenCLIP models."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        *,
        device: str | None = None,
        batch_size: int = 8,
        use_fp16: bool = True,
    ) -> None:
        requested_device: str
        if isinstance(device, torch.device):
            requested_device = device.type
        else:
            requested_device = str(device).strip().lower() if device is not None else "auto"
        if requested_device not in {"auto", "cuda", "cpu"}:
            logger.warning("OpenCLIP: unknown device '%s', defaulting to auto detection", requested_device)
            requested_device = "auto"

        requested_device = str(os.getenv("KOBATO_EMBED_DEVICE", device or "auto")).strip().lower()
        if requested_device == "auto":
            resolved_type = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda":
            if torch.cuda.is_available():
                resolved_type = "cuda"
            else:
                logger.warning("OpenCLIP: CUDA requested but not available; falling back to CPU")
                resolved_type = "cpu"
        else:
            resolved_type = "cpu"
        self._device = torch.device(resolved_type)
        self._batch_size = max(1, batch_size)

        available_pretrained = _available_pretrained_tags(model_name)
        resolved_pretrained, did_fallback = _select_pretrained_tag(pretrained, available_pretrained)
        if did_fallback:
            logger.warning(
                "Pretrained tag '%s' is not available for model '%s'; falling back to '%s'",
                (pretrained or ""),
                model_name,
                resolved_pretrained,
            )
        logger.info(
            "Loading OpenCLIP model '%s' with pretrained '%s' on %s",
            model_name,
            resolved_pretrained,
            self._device.type,
        )

        model, preprocess = _create_model_and_preprocess(model_name, resolved_pretrained, self._device)
        model.to(self._device)
        model.eval()

        self._use_fp16 = use_fp16 and self._device.type != "cpu"
        if self._use_fp16:
            model.half()

        self._model = model
        self._preprocess = preprocess
        # まずは素直に visual.output_dim を試す
        emb_dim = getattr(model.visual, "output_dim", None)
        if isinstance(emb_dim, (int,)):
            self._embedding_dim = int(emb_dim)
        else:
            # 次に text_projection（CLIP の線形射）から列数を拾う
            self._embedding_dim = None
            proj = getattr(model, "text_projection", None)
            try:
                if proj is not None and hasattr(proj, "shape"):
                    self._embedding_dim = int(proj.shape[1])
            except Exception:
                pass
            # それでも取れなければ、必要時に ensure_dim() で決める（遅延確定）

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of produced embeddings."""
        return self._embedding_dim

    @property
    def dim(self) -> int | None:
        return self._embedding_dim

    # ★ 追加：どうしても不明なとき 1 回だけ軽い推論で次元を確定する
    def ensure_dim(self) -> int:
        if self._embedding_dim is None:
            dummy = Image.new("RGB", (224, 224), (0, 0, 0))
            vec = self.embed_images([dummy])
            self._embedding_dim = int(vec.shape[1])
        return self._embedding_dim

    def _encode_tensor_batch(self, tensor_batch: torch.Tensor) -> torch.Tensor:
        if self._use_fp16 and tensor_batch.dtype != torch.float16:
            tensor_batch = tensor_batch.half()
        tensor_batch = tensor_batch.to(self._device)
        with torch.inference_mode():
            features = self._model.encode_image(tensor_batch)
        features = features.float()
        return torch.nn.functional.normalize(features, p=2, dim=1)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Encode a sequence of PIL images into normalized embedding vectors."""
        if not images:
            # ★ 次元未確定でもエラーにならないように（任意）
            if self._embedding_dim is None:
                return np.empty((0, 0), dtype=np.float32)
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        tensors = [self._preprocess(image) for image in images]
        embeddings: list[np.ndarray] = []
        for start in range(0, len(tensors), self._batch_size):
            batch = torch.stack(tensors[start : start + self._batch_size], dim=0)
            encoded = self._encode_tensor_batch(batch)
            embeddings.append(encoded.cpu().numpy().astype(np.float32))
        return np.concatenate(embeddings, axis=0)

    def embed_paths(self, paths: Sequence[str | Path]) -> tuple[np.ndarray, list[int]]:
        """Load images from ``paths`` and encode them, returning valid indices."""
        images: list[Image.Image] = []
        valid_indices: list[int] = []
        for index, path in enumerate(paths):
            image = safe_load_image(path)
            if image is None:
                continue
            images.append(image)
            valid_indices.append(index)
        embeddings = self.embed_images(images)
        return embeddings, valid_indices


__all__ = ["OpenClipEmbedder"]


def _available_pretrained_tags(model_name: str) -> list[str]:
    matches: list[str] = []
    for candidate_name, tag in open_clip.list_pretrained():
        if candidate_name.lower() == model_name.lower() and tag not in matches:
            matches.append(tag)
    return matches


def _select_pretrained_tag(requested: str, available: Sequence[str]) -> tuple[str, bool]:
    cleaned_requested = (requested or "").strip()
    if cleaned_requested and (not available or cleaned_requested in available):
        return cleaned_requested, False
    for candidate in ("laion2b_s32b_b82k", "openai"):
        if candidate in available:
            return candidate, cleaned_requested != candidate
    if available:
        chosen = available[0]
        return chosen, cleaned_requested != chosen
    fallback = cleaned_requested or "openai"
    return fallback, cleaned_requested != fallback


def _create_model_and_preprocess(
    model_name: str, pretrained_tag: str, device: torch.device
) -> tuple[torch.nn.Module, Any]:
    result = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag, device=device)
    try:
        length = len(result)
    except TypeError as exc:  # pragma: no cover - defensive safeguard
        raise RuntimeError("Unexpected return from open_clip.create_model_and_transforms") from exc
    if length == 3:
        model, preprocess_train, preprocess_val = result  # type: ignore[misc]
        preprocess = preprocess_val or preprocess_train
    elif length == 2:
        model, preprocess = result  # type: ignore[misc]
    else:
        raise RuntimeError("open_clip.create_model_and_transforms returned unexpected tuple length")
    return model, preprocess
