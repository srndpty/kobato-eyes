"""Utilities for parsing whitespace-delimited tag queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tagger.base import TagCategory


@dataclass(frozen=True)
class QueryFragment:
    """SQL fragment describing a WHERE clause and its parameters."""

    where: str
    params: list[object]


@dataclass(frozen=True)
class TagSpec:
    """Description of a single tag token extracted from a query."""

    raw: str
    category: Optional[str]
    name: str
    negative: bool = False


_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)

TAG_CHARS = r"[A-Za-z0-9:_'\-]+"
_TAG_VALID_RE = re.compile(f"^{TAG_CHARS}$")

_CATEGORY_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "0": "general",
    "general": "general",
    "1": "character",
    "character": "character",
    "2": "rating",
    "rating": "rating",
    "3": "copyright",
    "copyright": "copyright",
    "4": "artist",
    "artist": "artist",
    "5": "meta",
    "meta": "meta",
}

_CANONICAL_TO_CATEGORY: Dict[str, TagCategory] = {
    "general": TagCategory.GENERAL,
    "character": TagCategory.CHARACTER,
    "rating": TagCategory.RATING,
    "copyright": TagCategory.COPYRIGHT,
    "artist": TagCategory.ARTIST,
    "meta": TagCategory.META,
}


def tokenize_whitespace_only(query: str) -> List[str]:
    """Split ``query`` by Unicode whitespace without special quoting rules."""

    if not query:
        return []
    normalized = query.replace("\u3000", " ")
    return [token for token in _WHITESPACE_RE.split(normalized) if token]


def strip_negative_prefix(token: str) -> Tuple[bool, str]:
    """Return a tuple describing whether ``token`` is negative and its core value."""

    if token.startswith("-") and len(token) > 1:
        return True, token[1:]
    return False, token


def is_not_keyword(token: str) -> bool:
    """Return ``True`` if ``token`` is the NOT keyword (case insensitive)."""

    return token.upper() == "NOT"


def split_category(raw: str) -> Tuple[Optional[str], str]:
    """Split ``raw`` into an optional category and the remaining tag name."""

    if ":" not in raw:
        return None, raw
    left, right = raw.split(":", 1)
    if not left or not right:
        return None, raw
    canonical = _CATEGORY_ALIAS_TO_CANONICAL.get(left.lower())
    if canonical is None:
        return None, raw
    return canonical, right


def parse_search(query: str) -> Dict[str, List[TagSpec]]:
    """Parse ``query`` into positive and negative tag specifications."""

    tokens = tokenize_whitespace_only(query)
    include: List[TagSpec] = []
    exclude: List[TagSpec] = []
    free: List[TagSpec] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if is_not_keyword(token):
            if i + 1 < len(tokens):
                candidate = tokens[i + 1]
                _, core = strip_negative_prefix(candidate)
                category, name = split_category(core)
                spec = TagSpec(raw=core, category=category, name=name, negative=True)
                exclude.append(spec)
                i += 2
                continue
            free.append(TagSpec(raw=token, category=None, name=token))
            i += 1
            continue

        is_negative, core = strip_negative_prefix(token)
        if not core:
            free.append(TagSpec(raw=token, category=None, name=token, negative=is_negative))
            i += 1
            continue
        category, name = split_category(core)
        spec = TagSpec(raw=core, category=category, name=name, negative=is_negative)
        if is_negative:
            exclude.append(spec)
        else:
            include.append(spec)
        i += 1

    return {"include": include, "exclude": exclude, "free": free}


def _resolve_category_key(category: Optional[str]) -> Optional[int]:
    """Convert a canonical category label into its database key."""

    if category is None:
        return None
    lookup = _CANONICAL_TO_CATEGORY.get(category.lower())
    if lookup is None:
        return None
    return int(lookup)


def _exists_clause_for_tag(alias_file: str, spec: TagSpec) -> Tuple[str, List[object]]:
    """Return an EXISTS clause and parameters for ``spec`` against ``alias_file``."""

    category_key = _resolve_category_key(spec.category)
    if category_key is not None:
        clause = (
            "EXISTS (SELECT 1 FROM file_tags ft "
            "JOIN tags t ON t.id = ft.tag_id "
            f"WHERE ft.file_id = {alias_file}.id "
            "AND t.category = ? AND t.name = ?)"
        )
        return clause, [category_key, spec.name]
    clause = (
        "EXISTS (SELECT 1 FROM file_tags ft "
        "JOIN tags t ON t.id = ft.tag_id "
        f"WHERE ft.file_id = {alias_file}.id "
        "AND t.name = ?)"
    )
    return clause, [spec.name]


def translate_query(query: str, *, file_alias: str = "f") -> QueryFragment:
    """Convert ``query`` into a QueryFragment for database filtering."""

    parsed = parse_search(query)
    clauses: List[str] = []
    params: List[object] = []

    for spec in parsed["include"]:
        clause, clause_params = _exists_clause_for_tag(file_alias, spec)
        clauses.append(clause)
        params.extend(clause_params)

    for spec in parsed["exclude"]:
        clause, clause_params = _exists_clause_for_tag(file_alias, spec)
        clauses.append(f"NOT {clause}")
        params.extend(clause_params)

    if not clauses:
        return QueryFragment(where="1=1", params=[])

    where = " AND ".join(clauses)
    return QueryFragment(where=where, params=params)


def extract_positive_tag_terms(query: str) -> List[str]:
    """Return lower-cased unique tag names that appear positively in ``query``."""

    parsed = parse_search(query)
    seen: set[str] = set()
    ordered: List[str] = []
    for spec in parsed["include"]:
        name = spec.name.strip()
        if not name:
            continue
        if _TAG_VALID_RE.fullmatch(name) is None:
            continue
        lowered = name.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered.append(lowered)
    return ordered


__all__ = [
    "QueryFragment",
    "TAG_CHARS",
    "TagSpec",
    "extract_positive_tag_terms",
    "parse_search",
    "tokenize_whitespace_only",
    "translate_query",
]
