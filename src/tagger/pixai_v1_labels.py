"""Validated label maps for PixAI v1 ONNX distributions."""

from __future__ import annotations

import json
from pathlib import Path

from tagger.base import TagCategory
from tagger.labels_util import TagMeta

LABEL_COUNT = 30877
DEFAULT_THRESHOLDS = dict(general=0.17, character=0.27, copyright=0.24, style=0.15, meta=0.17, rating=0.41)
CATEGORY_COUNTS = dict(general=15043, character=8308, copyright=2460, style=4917, meta=145, rating=4)
PREPROCESS_VERSION = "pixai-v1-float-bilinear-antialias-pad-1008-v1"


def resolve_labels_json(model_path: str | Path, labels_json: str | Path | None = None) -> Path:
    """Find an explicit label map or an adjacent supported distribution file."""
    if labels_json:
        path = Path(labels_json).expanduser()
        if not path.is_file():
            raise ValueError(f"PixAI v1 labels not found: {path}")
        return path
    for name in ("tags.json", "config.json"):
        path = Path(model_path).expanduser().parent / name
        if path.is_file():
            return path
    raise ValueError("PixAI v1 requires tags.json or config.json next to the model")


def load_labels_json(path: Path) -> list[TagMeta]:
    """Read noaione tags.json or upstream/bdsqlsz config.json in output order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("num_classes") != LABEL_COUNT:
        raise ValueError(f"PixAI v1 label map must contain {LABEL_COUNT} classes")
    groups = data.get("categories")
    if groups is None:
        tags = data.get("tags")
        if not isinstance(tags, list):
            raise ValueError("PixAI v1 config requires an ordered tags list")
        groups = []
        offset = 0
        for name, count in data.get("tags_split", []):
            groups.append(dict(name=name, count=count, offset=offset, tags=tags[offset : offset + count]))
            offset += count
        if offset != len(tags):
            raise ValueError("PixAI v1 tags_split does not cover the tags list")
    if not isinstance(groups, list):
        raise ValueError("PixAI v1 categories must be a list")
    labels: list[TagMeta | None] = [None] * LABEL_COUNT
    seen: set[str] = set()
    names: set[str] = set()
    for group in groups:
        name, count, offset, tags = (group[key] for key in ("name", "count", "offset", "tags"))
        if name in seen or name not in CATEGORY_COUNTS or count != CATEGORY_COUNTS[name]:
            raise ValueError(f"Invalid PixAI v1 category: {name}")
        if not isinstance(offset, int) or offset < 0 or offset + count > LABEL_COUNT:
            raise ValueError(f"Invalid PixAI v1 offset: {name}")
        if not isinstance(tags, list) or len(tags) != count:
            raise ValueError(f"Invalid PixAI v1 tag count: {name}")
        seen.add(name)
        for index, tag in enumerate(tags, offset):
            if not isinstance(tag, str) or not tag or tag in names or labels[index] is not None:
                raise ValueError(f"Invalid or overlapping PixAI v1 label at {index}")
            labels[index] = TagMeta(tag, int(TagCategory[name.upper()]))
            names.add(tag)
    if seen != set(CATEGORY_COUNTS) or any(label is None for label in labels):
        raise ValueError("PixAI v1 categories do not cover all model outputs")
    return [label for label in labels if label is not None]
