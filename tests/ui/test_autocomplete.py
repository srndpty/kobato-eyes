from __future__ import annotations

import pytest

from ui.autocomplete import extract_completion_token, replace_completion_token


DELIMITERS = " \t\r\n,;()"


@pytest.mark.parametrize(
    ("text", "cursor", "expected"),
    [
        ("tag1 AND ha", None, ("ha", 9, 11)),
        ("tag1 AND ha", 10, ("ha", 9, 11)),
        ("tag1, ta", None, ("ta", 6, 8)),
        ("(", None, ("", 1, 1)),
        ("category:ge", None, ("category:ge", 0, 11)),
    ],
)
def test_extract_completion_token(text: str, cursor: int | None, expected: tuple[str, int, int]) -> None:
    token, start, end = extract_completion_token(text, cursor)
    assert (token, start, end) == expected


@pytest.mark.parametrize(
    ("text", "cursor", "replacement", "expected_text"),
    [
        ("tag1 AND ha", None, "hatsune_miku", "tag1 AND hatsune_miku "),
        ("(ta)", 3, "hatsune_miku", "(hatsune_miku)"),
        ("category:ge", None, "category:general", "category:general "),
        ("tag1 AND ta more", 10, "hatsune_miku", "tag1 AND hatsune_miku more"),
    ],
)
def test_replace_completion_token(text: str, cursor: int | None, replacement: str, expected_text: str) -> None:
    _, start, end = extract_completion_token(text, cursor)
    new_text, cursor_pos = replace_completion_token(text, start, end, replacement)
    assert new_text == expected_text
    suffix = text[end:]
    expected_cursor = start + len(replacement)
    needs_space = (not suffix) or suffix[0] not in DELIMITERS
    if replacement and needs_space and not replacement.endswith(" "):
        expected_cursor += 1
    assert cursor_pos == expected_cursor
