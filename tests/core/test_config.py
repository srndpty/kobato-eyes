"""Tests for persistent configuration handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import config_path, load_settings, save_settings
from core.settings import PipelineSettings


def test_config_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = config_path()
    if path.exists():
        path.unlink()

    settings = load_settings()
    assert settings.roots == []

    updated = PipelineSettings(
        roots=[tmp_path / "root"],
        excluded=[tmp_path / "skip"],
        hamming_threshold=5,
        cosine_threshold=0.15,
        ssim_threshold=0.95,
        model_name="ViT-H-14",
    )
    save_settings(updated)

    reloaded = load_settings()
    assert reloaded.roots == updated.roots
    assert reloaded.excluded == updated.excluded
    assert reloaded.hamming_threshold == 5
    assert reloaded.cosine_threshold == pytest.approx(0.15)
    assert reloaded.model_name == "ViT-H-14"
