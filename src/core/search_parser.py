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


@dataclass(frozen=True)
class SearchToken:
    """Token produced from a search expression for testing/debugging."""

    raw: str
    spec: TagSpec | None
    is_negative: bool
    is_free: bool


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


def tokenize_search(expr: str) -> list[SearchToken]:
    """Tokenize ``expr`` highlighting tag/free segments and negation."""

    tokens = shlex.split(expr, posix=True) if expr else []
    results: list[SearchToken] = []

    skip_next = False
    pending_negation = False
    pending_token: str | None = None

    for raw in tokens:
        if not raw:
            continue
        if skip_next:
            results.append(SearchToken(raw=raw, spec=None, is_negative=False, is_free=True))
            skip_next = False
            continue
        if raw == "-":
            results.append(SearchToken(raw=raw, spec=None, is_negative=False, is_free=True))
            skip_next = True
            pending_negation = False
            pending_token = None
            continue

        if raw.upper() == "NOT":
            pending_negation = True
            pending_token = raw
            continue

        negative = False
        candidate = raw
        if raw.startswith("-") and len(raw) > 1:
            negative = True
            candidate = raw[1:]

        spec = _parse_tag(candidate)
        if spec is None:
            if pending_negation and pending_token:
                results.append(SearchToken(raw=pending_token, spec=None, is_negative=False, is_free=True))
            results.append(SearchToken(raw=raw, spec=None, is_negative=False, is_free=True))
            pending_negation = False
            pending_token = None
            continue

        is_negative = negative or pending_negation
        results.append(SearchToken(raw=raw, spec=spec, is_negative=is_negative, is_free=False))
        pending_negation = False
        pending_token = None

    if pending_negation and pending_token:
        results.append(SearchToken(raw=pending_token, spec=None, is_negative=False, is_free=True))

    return results


def parse_search(expr: str) -> dict[str, list[TagSpec] | list[str]]:
    """Parse ``expr`` into tag filters and free-text terms."""

    tokens = tokenize_search(expr)

    include: list[TagSpec] = []
    exclude: list[TagSpec] = []
    free: list[str] = []

    for token in tokens:
        if token.spec is not None:
            if token.is_negative:
                exclude.append(token.spec)
            else:
                include.append(token.spec)
            continue
        if token.is_free:
            free.append(token.raw)

    return {"include": include, "exclude": exclude, "free": free}


__all__ = ["TagSpec", "SearchToken", "tokenize_search", "parse_search"]
