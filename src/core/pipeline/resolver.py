from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from core.config import PipelineSettings
from tagger.base import ITagger, TagCategory

from .utils import _serialise_max_tags, _serialise_thresholds, detect_tagger_provider

logger = logging.getLogger(__name__)


def _resolve_tagger(
    settings: PipelineSettings,
    override: ITagger | None,
    *,
    thresholds: Mapping[TagCategory, float] | None = None,
    max_tags: Mapping[TagCategory, int] | None = None,
) -> ITagger:
    serialised_thresholds = _serialise_thresholds(thresholds)
    serialised_max_tags = _serialise_max_tags(max_tags)

    if override is not None:
        name = type(override).__name__
        model_path = getattr(override, "model_path", None)
        tags_csv = getattr(override, "tags_csv", None)
        provider_value = getattr(override, "provider", None)
        logger.info(
            "Tagger in use: %s, provider=%s, model=%s, tags_csv=%s, thresholds=%s, max_tags=%s",
            name,
            provider_value,
            model_path,
            tags_csv,
            serialised_thresholds,
            serialised_max_tags,
        )
        return override

    tagger_name = settings.tagger.name
    lowered = tagger_name.lower()
    model_path_value = settings.tagger.model_path
    tags_csv_value = getattr(settings.tagger, "tags_csv", None)
    provider = detect_tagger_provider(settings)

    if lowered == "dummy":
        from tagger.dummy import DummyTagger

        tagger_instance: ITagger = DummyTagger()
    elif lowered == "wd14-onnx":
        if not settings.tagger.model_path:
            raise ValueError("WD14: model_path is required")
        model_path_obj = Path(settings.tagger.model_path)
        if provider == "pixai":
            from tagger.pixai_onnx import PixaiOnnxTagger

            tagger_instance = PixaiOnnxTagger(model_path_obj, tags_csv=settings.tagger.tags_csv)
        else:
            from tagger.wd14_onnx import WD14Tagger

            tagger_instance = WD14Tagger(model_path_obj, tags_csv=settings.tagger.tags_csv)
        model_path_value = str(model_path_obj)
    else:
        raise ValueError(f"Unknown tagger '{settings.tagger.name}'")

    logger.info(
        "Tagger in use: %s, provider=%s, model=%s, tags_csv=%s, thresholds=%s, max_tags=%s",
        tagger_name,
        provider,
        model_path_value,
        tags_csv_value,
        serialised_thresholds,
        serialised_max_tags,
    )
    return tagger_instance


__all__ = ["_resolve_tagger"]
