"""Tests for tagger resolution branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import PipelineSettings
from core.pipeline.resolver import _resolve_tagger
from tagger.base import TagCategory
from tagger.dummy import DummyTagger


class _OverrideTagger:
    """Small concrete tagger stand-in used for override resolution."""

    provider = "override-provider"
    model_path = Path("override.onnx")
    tags_csv = Path("tags.csv")

    def prepare_batch_from_rgb_np(self, images):  # type: ignore[no-untyped-def]
        return images

    def infer_batch_prepared(self, batch, *, thresholds=None, max_tags=None):  # type: ignore[no-untyped-def]
        return []

    def infer_batch(self, images, *, thresholds=None, max_tags=None):  # type: ignore[no-untyped-def]
        return []


def test_resolve_tagger_returns_dummy_instance() -> None:
    settings = PipelineSettings.from_mapping({"tagger": {"name": "dummy"}})

    tagger, thresholds, max_tags = _resolve_tagger(settings, None)

    assert isinstance(tagger, DummyTagger)
    assert thresholds is None
    assert max_tags is None


def test_resolve_tagger_rejects_unknown_name() -> None:
    settings = PipelineSettings.from_mapping({"tagger": {"name": "unknown"}})

    with pytest.raises(ValueError, match="Unknown tagger"):
        _resolve_tagger(settings, None)


def test_resolve_tagger_requires_wd14_model_path() -> None:
    settings = PipelineSettings.from_mapping({"tagger": {"name": "wd14-onnx"}})

    with pytest.raises(ValueError, match="model_path is required"):
        _resolve_tagger(settings, None)


def test_resolve_tagger_applies_pixai_defaults_for_override() -> None:
    settings = PipelineSettings.from_mapping({"tagger": {"name": "dummy", "provider": "pixai"}})
    override = _OverrideTagger()

    tagger, thresholds, max_tags = _resolve_tagger(settings, override)

    assert tagger is override
    assert thresholds == {
        TagCategory.GENERAL: 0.4,
        TagCategory.CHARACTER: 0.8,
        TagCategory.COPYRIGHT: 0.8,
    }
    assert max_tags == {
        TagCategory.GENERAL: 128,
        TagCategory.CHARACTER: 10,
        TagCategory.COPYRIGHT: 10,
    }


def test_resolve_tagger_preserves_explicit_override_limits() -> None:
    settings = PipelineSettings.from_mapping({"tagger": {"name": "dummy", "provider": "pixai"}})
    override = _OverrideTagger()
    thresholds = {TagCategory.GENERAL: 0.55}
    max_tags = {TagCategory.GENERAL: 7}

    tagger, resolved_thresholds, resolved_max_tags = _resolve_tagger(
        settings,
        override,
        thresholds=thresholds,
        max_tags=max_tags,
    )

    assert tagger is override
    assert resolved_thresholds is thresholds
    assert resolved_max_tags is max_tags
