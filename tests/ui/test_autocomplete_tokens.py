from __future__ import annotations

from ui.autocomplete import extract_completion_token, replace_completion_token


def test_extract_completion_token_basic() -> None:
    text = "solo AND ch"
    token, start, end = extract_completion_token(text)
    assert token == "ch"
    assert text[start:end] == "ch"


def test_extract_completion_token_with_cursor() -> None:
    text = "solo AND ch"
    token, start, end = extract_completion_token(text, cursor_position=4)
    assert token == "solo"
    assert (start, end) == (0, 4)


def test_replace_completion_token() -> None:
    text = "solo AND ch"
    token, start, end = extract_completion_token(text)
    assert token == "ch"
    new_text, cursor = replace_completion_token(text, start, end, "character:")
    assert new_text == "solo AND character: "
    assert cursor == len(new_text)
