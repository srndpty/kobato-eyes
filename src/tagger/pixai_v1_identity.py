"""Content identities for PixAI v1 retagging decisions."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.schema import TaggerSettings

from tagger.base import MaxTagsMap, ThresholdMap
from tagger.pixai_v1_labels import DEFAULT_THRESHOLDS, PREPROCESS_VERSION, resolve_labels_json


@lru_cache(maxsize=32)
def _hash_file(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """Hash large files once per filesystem identity, without loading them whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(path: Path) -> str:
    """Return a stable content hash; represent absent files without breaking settings."""
    if not path.is_file():
        return "missing:" + str(path)
    stat = path.stat()
    return _hash_file(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def v1_signature(settings: TaggerSettings, *, thresholds: ThresholdMap | None, max_tags: MaxTagsMap | None) -> str:
    """Include ONNX, external weights, label order, preprocessing and effective options."""
    from tagger.base import TagCategory

    model = Path(settings.model_path or "__missing_model__").expanduser()
    try:
        labels = resolve_labels_json(model, settings.labels_json)
    except ValueError:
        labels = model.parent / "tags.json"
    # Both supported distributions either use model.onnx.data or embed weights.
    weights = model.with_name(model.name + ".data")
    threshold_values = settings.thresholds if thresholds is None else {c.name.lower(): v for c, v in thresholds.items()}
    threshold_values = dict(DEFAULT_THRESHOLDS) | dict(threshold_values)
    payload = dict(
        model=content_digest(model),
        weights=content_digest(weights) if weights.is_file() else "embedded",
        labels=content_digest(labels),
        preprocess=PREPROCESS_VERSION,
        thresholds={str(k): float(v) for k, v in threshold_values.items()},
        max_tags={TagCategory(k).name.lower(): int(v) for k, v in (max_tags or {}).items()},
    )
    return "pixai-v1-onnx:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
