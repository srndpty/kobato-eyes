"""Tests for pipeline logging helpers."""

from __future__ import annotations

import logging

from core.pipeline import _build_max_tags_map, _build_threshold_map, _resolve_tagger
from core.settings import PipelineSettings, TaggerSettings


def test_resolve_tagger_logs_active_choice(caplog) -> None:
    settings = PipelineSettings(tagger=TaggerSettings(name="dummy"))
    caplog.set_level(logging.INFO, logger="core.pipeline")
    thresholds = _build_threshold_map(settings.tagger.thresholds)
    max_tags = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))

    _resolve_tagger(settings, None, thresholds=thresholds, max_tags=max_tags)

    assert any(
        record.message.startswith("Tagger in use: dummy") for record in caplog.records
    )
