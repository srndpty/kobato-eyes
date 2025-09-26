"""Pure utility helpers for the tag autocomplete logic."""

from __future__ import annotations

_DELIMITERS = " \t\r\n,;:"


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
    """Return the token under the cursor and its range."""

    if cursor_position is None:
        cursor_position = len(text)
    cursor_position = max(0, min(cursor_position, len(text)))
    if not text:
        return "", cursor_position, cursor_position

    start = cursor_position
    while start > 0 and text[start - 1] not in _DELIMITERS:
        start -= 1

    end = cursor_position
    while end < len(text) and text[end] not in _DELIMITERS:
        end += 1

    return text[start:end], start, end


def replace_completion_token(
    text: str, start: int, end: int, replacement: str
) -> tuple[str, int]:
    """Replace the substring in ``text`` and return the new text and cursor."""

    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))

    suffix = text[end:]
    insertion = replacement
    if replacement:
        needs_space = (not suffix) or (suffix[0] not in _DELIMITERS)
        if needs_space and (not replacement.endswith(" ")):
            insertion = f"{replacement} "

    new_text = f"{text[:start]}{insertion}{suffix}"
    new_cursor = start + len(insertion)
    return new_text, new_cursor


__all__ = ["abbreviate_count", "extract_completion_token", "replace_completion_token"]

