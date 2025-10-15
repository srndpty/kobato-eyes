"""Tests for pipeline logging helpers."""

from __future__ import annotations

import logging

from core.config import PipelineSettings, TaggerSettings
from core.pipeline import _build_max_tags_map, _build_threshold_map, _resolve_tagger


def test_resolve_tagger_logs_active_choice(caplog) -> None:
    settings = PipelineSettings(tagger=TaggerSettings(name="dummy"))
    caplog.set_level(logging.INFO, logger="core.pipeline")
    thresholds = _build_threshold_map(settings.tagger.thresholds)
    max_tags = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))

    _resolve_tagger(settings, None, thresholds=thresholds, max_tags=max_tags)

    assert any("Tagger in use: dummy" in record.message for record in caplog.records)
