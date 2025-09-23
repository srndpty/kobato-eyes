"""Embedding utilities built on top of OpenCLIP."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

from utils.image_io import safe_load_image


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
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._batch_size = max(1, batch_size)

        model, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.to(self._device)
        model.eval()

        self._use_fp16 = use_fp16 and self._device.type != "cpu"
        if self._use_fp16:
            model.half()

        self._model = model
        self._preprocess = preprocess
        self._embedding_dim = int(getattr(model.visual, "output_dim"))

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of produced embeddings."""
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
            return np.empty((0, self.embedding_dim), dtype=np.float32)

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
