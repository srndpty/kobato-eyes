"""Parse lightweight tag search expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_TAG_BODY = r"[A-Za-z0-9:_\-!?><.@()^+]+"
_BOUNDARY = r"(?:^|(?<=\s))"
_NEGATIVE_RE = re.compile(rf"{_BOUNDARY}-(?:\"(?P<neg_q>[^\"]+)\"|(?P<neg>{_TAG_BODY}))")
_POSITIVE_RE = re.compile(rf"{_BOUNDARY}(?:\"(?P<pos_q>[^\"]+)\"|(?P<pos>{_TAG_BODY}))")
_VALID_TAG = re.compile(rf"^{_TAG_BODY}$")

_CATEGORY_ALIASES: dict[str, str] = {
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


@dataclass(frozen=True)
class TagSpec:
    """Single tag term extracted from a user query."""

    raw: str
    category: str | None
    name: str


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    raw: str
    quoted: bool


def _mark_used(flags: list[bool], start: int, end: int) -> None:
    for index in range(start, min(end, len(flags))):
        flags[index] = True


def _split_category(tag: str) -> tuple[str | None, str]:
    if ":" not in tag:
        return None, tag
    prefix, remainder = tag.split(":", 1)
    if not remainder:
        return None, tag
    alias = _CATEGORY_ALIASES.get(prefix.lower())
    if alias is None:
        return None, tag
    return alias, remainder


def _create_spec(raw: str) -> TagSpec | None:
    if not raw or raw == "-":
        return None
    if raw.upper() == "NOT":
        return None
    if not _VALID_TAG.fullmatch(raw):
        return None
    category, name = _split_category(raw)
    return TagSpec(raw=raw, category=category, name=name)


def _invalid_negative_spans(expr: str) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    length = len(expr)
    index = 0
    while index < length:
        if expr[index] == "-" and (index + 1 >= length or expr[index + 1].isspace()):
            follower = index + 1
            while follower < length and expr[follower].isspace():
                follower += 1
            if follower < length:
                end = follower
                while end < length and not expr[end].isspace():
                    end += 1
                spans.add((follower, end))
        index += 1
    return spans


def _collect_free(expr: str, used: Sequence[bool]) -> list[str]:
    free: list[str] = []
    length = len(expr)
    index = 0
    while index < length:
        if expr[index].isspace():
            index += 1
            continue
        start = index
        while index < length and not expr[index].isspace():
            index += 1
        end = index
        if not any(used[start:end]):
            free.append(expr[start:end])
    return free


def parse_search(expr: str) -> dict[str, list[TagSpec] | list[str]]:
    """Parse ``expr`` into tag filters and free-text segments."""

    include: list[TagSpec] = []
    exclude: list[TagSpec] = []
    used = [False] * len(expr)

    invalid_spans = _invalid_negative_spans(expr)

    for match in _NEGATIVE_RE.finditer(expr):
        start, end = match.span()
        tag = match.group("neg_q") or match.group("neg")
        spec = _create_spec(tag)
        if spec is None:
            continue
        exclude.append(spec)
        _mark_used(used, start, end)

    candidates: list[_Candidate] = []
    for match in _POSITIVE_RE.finditer(expr):
        start, end = match.span()
        if any(used[start:end]):
            continue
        if (start, end) in invalid_spans:
            continue
        raw = match.group("pos_q") or match.group("pos") or ""
        quoted = match.group("pos_q") is not None
        candidates.append(_Candidate(start=start, end=end, raw=raw, quoted=quoted))

    consumed = [False] * len(candidates)
    index = 0
    while index < len(candidates):
        if consumed[index]:
            index += 1
            continue
        token = candidates[index]
        raw_upper = token.raw.upper()
        if not token.quoted and raw_upper == "NOT":
            lookahead = index + 1
            while lookahead < len(candidates) and consumed[lookahead]:
                lookahead += 1
            if lookahead < len(candidates):
                target = candidates[lookahead]
                spec = _create_spec(target.raw)
                if spec is not None:
                    exclude.append(spec)
                    _mark_used(used, token.start, token.end)
                    _mark_used(used, target.start, target.end)
                    consumed[index] = True
                    consumed[lookahead] = True
                    index += 1
                    continue
        spec = _create_spec(token.raw)
        if spec is not None:
            include.append(spec)
            _mark_used(used, token.start, token.end)
            consumed[index] = True
        index += 1

    free = _collect_free(expr, used)
    return {"include": include, "exclude": exclude, "free": free}


def identify_token_at(text: str, cursor: int) -> tuple[int, int, str]:
    """Return the token bounds and text at ``cursor`` within ``text``."""

    length = len(text)
    cursor = max(0, min(cursor, length))
    if length == 0:
        return 0, 0, ""

    start = cursor
    if cursor > 0 and (cursor == length or not text[cursor].isspace()):
        while start > 0 and not text[start - 1].isspace():
            start -= 1

    end = cursor
    while end < length and not text[end].isspace():
        end += 1

    token = text[start:end]
    if token and cursor < length and text[cursor].isspace():
        token = ""
        start = cursor
        end = cursor
    return start, end, token


def strip_negative_prefix(token: str) -> tuple[bool, str, bool]:
    """Return minus flag, core text and quote flag for ``token``."""

    if not token:
        return False, "", False

    negative = False
    core = token
    if core.startswith("-"):
        negative = True
        core = core[1:]

    quoted = False
    if core:
        first = core[0]
        if first in {'"', "'"}:
            quoted = True
            if len(core) >= 2 and core[-1] == first:
                core = core[1:-1]
            else:
                core = core[1:]

    return negative, core, quoted


__all__ = [
    "TagSpec",
    "parse_search",
    "identify_token_at",
    "strip_negative_prefix",
]
