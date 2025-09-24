"""Pure utility helpers for the tag autocomplete logic."""

from __future__ import annotations

import re

_LAST_TOKEN_RE = re.compile(r"([A-Za-z0-9_:+><=.]+)$")


def abbreviate_count(value: object) -> str:
    """Return a compact textual representation of a popularity count."""

    try:
        count = int(value) if value is not None else 0
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return ""
    thresholds = (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "k"),
    )
    for threshold, suffix in thresholds:
        if count >= threshold:
            quotient = count / threshold
            if quotient >= 100:
                formatted = f"{quotient:.0f}"
            elif quotient >= 10:
                formatted = f"{quotient:.1f}"
            else:
                formatted = f"{quotient:.2f}"
            formatted = formatted.rstrip("0").rstrip(".")
            return f"{formatted}{suffix}"
    return str(count)


def extract_completion_token(
    text: str, cursor_position: int | None = None
) -> tuple[str, int, int]:
    """Return the trailing token within ``text`` and its range."""

    if cursor_position is None:
        cursor_position = len(text)
    cursor_position = max(0, min(cursor_position, len(text)))
    prefix = text[:cursor_position]
    match = _LAST_TOKEN_RE.search(prefix)
    if not match:
        return "", cursor_position, cursor_position
    start = match.start(1)
    end = match.end(1)
    return match.group(1), start, end


def replace_completion_token(
    text: str, start: int, end: int, replacement: str
) -> tuple[str, int]:
    """Replace the substring in ``text`` and return the new text and cursor."""

    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    new_text = f"{text[:start]}{replacement}{text[end:]}"
    new_cursor = start + len(replacement)
    return new_text, new_cursor


__all__ = ["abbreviate_count", "extract_completion_token", "replace_completion_token"]

