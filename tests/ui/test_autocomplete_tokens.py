from __future__ import annotations

from pathlib import Path

from tagger.labels_util import load_selected_tags
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
    assert new_text == "solo AND character:"
    assert cursor == len(new_text)


def test_load_selected_tags_single_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_single.csv"
    csv_path.write_text("# comment\nname\n1girl\n\n", encoding="utf-8")
    tags = load_selected_tags(csv_path)
    assert tags == [("1girl", 0)]


def test_load_selected_tags_two_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_two.csv"
    csv_path.write_text("123,1girl\ncharacter:kobato,character\n", encoding="utf-8")
    tags = load_selected_tags(csv_path)
    assert ("1girl", 0) in tags
    assert ("character:kobato", 1) in tags


def test_load_selected_tags_four_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_four.csv"
    csv_path.write_text("1,solo,0,1000\n2,artist:name,4,50\n", encoding="utf-8")
    tags = load_selected_tags(csv_path)
    assert ("solo", 0) in tags
    assert ("artist:name", 4) in tags
