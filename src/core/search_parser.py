"""Parse lightweight tag search expressions."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Mapping

from tagger.base import TagCategory

_TAG_PATTERN = re.compile(r"^[A-Za-z0-9:_\-!?><.@()^+]+$")

_CATEGORY_ALIASES: Mapping[str, TagCategory] = {
    "0": TagCategory.GENERAL,
    "general": TagCategory.GENERAL,
    "1": TagCategory.CHARACTER,
    "character": TagCategory.CHARACTER,
    "2": TagCategory.RATING,
    "rating": TagCategory.RATING,
    "3": TagCategory.COPYRIGHT,
    "copyright": TagCategory.COPYRIGHT,
    "4": TagCategory.ARTIST,
    "artist": TagCategory.ARTIST,
    "5": TagCategory.META,
    "meta": TagCategory.META,
}


@dataclass(frozen=True)
class TagSpec:
    """Single tag term extracted from a user query."""

    name: str
    category: TagCategory | None = None


def _parse_tag(text: str) -> TagSpec | None:
    if not text or text == "-":
        return None
    if not _TAG_PATTERN.fullmatch(text):
        return None
    category: TagCategory | None = None
    if ":" in text:
        prefix, rest = text.split(":", 1)
        if rest:
            alias = prefix.lower()
            category = _CATEGORY_ALIASES.get(alias)
    return TagSpec(name=text, category=category)


def parse_search(expr: str) -> dict[str, list[TagSpec] | list[str]]:
    """Parse ``expr`` into tag filters and free-text terms."""

    tokens = shlex.split(expr, posix=True) if expr else []
    include: list[TagSpec] = []
    exclude: list[TagSpec] = []
    free: list[str] = []

    skip_next = False
    for raw in tokens:
        if not raw:
            continue
        if skip_next:
            free.append(raw)
            skip_next = False
            continue
        if raw == "-":
            free.append(raw)
            skip_next = True
            continue
        if raw.startswith("-") and len(raw) > 1:
            candidate = raw[1:]
            spec = _parse_tag(candidate)
            if spec is not None:
                exclude.append(spec)
            else:
                free.append(raw)
            continue
        spec = _parse_tag(raw)
        if spec is not None:
            include.append(spec)
        else:
            free.append(raw)

    return {"include": include, "exclude": exclude, "free": free}


__all__ = ["TagSpec", "parse_search"]
