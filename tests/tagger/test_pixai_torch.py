"""Smoke tests for the PixAI PyTorch tagger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tagger.base import TagCategory

pytest.importorskip("torch")
pytest.importorskip("timm")


def _write_pixai_assets(tmp_path: Path, num_classes: int) -> None:
    import torch

    weights_path = tmp_path / "model_v0.9.pth"
    head_weight = torch.zeros((num_classes, 1024), dtype=torch.float32)
    head_bias = torch.full((num_classes,), 2.5, dtype=torch.float32)  # sigmoid -> ~0.924
    torch.save({"head.weight": head_weight, "head.bias": head_bias}, weights_path)

    tags = [
        {"name": "blue_sky", "category": "general", "index": 0},
        {"name": "sakura_haruno", "category": "character", "index": 1},
    ]
    (tmp_path / "tags_v0.9_13k.json").write_text(json.dumps(tags), encoding="utf-8")
    mapping = {"sakura_haruno": ["naruto"]}
    (tmp_path / "char_ip_map.json").write_text(json.dumps(mapping), encoding="utf-8")


def test_pixai_tagger_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from torch import nn

    from tagger.pixai_torch import PixAITagger

    class _DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def reset_classifier(self, num_classes: int, **_: object) -> None:  # pragma: no cover - behaviour is simple
            self.num_classes = num_classes

        def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
            batch = inputs.shape[0]
            return torch.ones((batch, 1024), dtype=inputs.dtype, device=inputs.device)

    monkeypatch.setattr("tagger.pixai_torch.timm.create_model", lambda *_args, **_kw: _DummyEncoder())

    _write_pixai_assets(tmp_path, num_classes=2)

    tagger = PixAITagger(
        tmp_path,
        default_thresholds={
            TagCategory.GENERAL: 0.5,
            TagCategory.CHARACTER: 0.5,
            TagCategory.COPYRIGHT: 0.5,
        },
    )

    arrays = [
        np.full((448, 448, 3), 128, dtype=np.float32),
        np.full((448, 448, 3), 64, dtype=np.float32),
    ]
    prepared = tagger.prepare_batch_from_rgb_np(arrays)
    assert prepared.shape == (2, 448, 448, 3)
    assert prepared.dtype == np.float32

    results = tagger.infer_batch_prepared(prepared)
    assert len(results) == 2

    for result in results:
        general = {pred.name for pred in result.tags if pred.category == TagCategory.GENERAL}
        characters = {pred.name for pred in result.tags if pred.category == TagCategory.CHARACTER}
        copyrights = {pred.name for pred in result.tags if pred.category == TagCategory.COPYRIGHT}
        assert "blue_sky" in general
        assert "sakura_haruno" in characters
        assert "naruto" in copyrights
