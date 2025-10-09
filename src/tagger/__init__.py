"""Tagging pipelines for kobato-eyes."""

from __future__ import annotations

from typing import Any

from tagger.base import ITagger


def create_tagger(kind: str, **kwargs: Any) -> ITagger:
    """Instantiate a tagger implementation by name."""

    lowered = kind.strip().lower()
    if lowered == "dummy":
        from tagger.dummy import DummyTagger

        return DummyTagger(**kwargs)
    if lowered in {"wd14", "wd14-onnx"}:
        from tagger.wd14_onnx import WD14Tagger

        return WD14Tagger(**kwargs)
    if lowered == "pixai":
        from tagger.pixai_torch import PixAITagger

        return PixAITagger(**kwargs)
    raise ValueError(f"Unknown tagger '{kind}'")


__all__ = ["create_tagger"]
