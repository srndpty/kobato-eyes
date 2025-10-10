"""Tests for default and legacy pipeline settings behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import PipelineSettings


def test_defaults_are_applied() -> None:
    settings = PipelineSettings()

    assert settings.hamming_threshold == 10
    assert settings.batch_size == 8
    assert settings.ssim_threshold == pytest.approx(0.92)
    assert settings.allow_exts == {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tiff",
    }
    assert settings.tagger.thresholds == {
        "general": pytest.approx(0.35),
        "character": pytest.approx(0.25),
        "copyright": pytest.approx(0.25),
    }


def test_legacy_mapping_fills_missing_fields(tmp_path: Path) -> None:
    legacy = {
        "roots": [str(tmp_path / "images")],
        "excludes": ["C:/Windows"],
        "allow_exts": ["PNG", ".webp"],
        "tagger": {"thresholds": {"general": 0.5}},
    }

    settings = PipelineSettings.from_mapping(legacy)

    assert settings.roots == [str((tmp_path / "images").expanduser())]
    assert any(path.endswith("Windows") for path in settings.excluded)
    assert settings.allow_exts == {".png", ".webp"}
    assert settings.hamming_threshold == 10
    assert settings.ssim_threshold == pytest.approx(0.92)
    assert settings.tagger.thresholds["general"] == pytest.approx(0.5)
    assert settings.tagger.thresholds["character"] == pytest.approx(0.25)
    assert settings.tagger.thresholds["copyright"] == pytest.approx(0.25)
