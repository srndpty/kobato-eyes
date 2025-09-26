from __future__ import annotations

import pytest

# 実装と同じデリミタを使う（将来変更してもテストが追従する）
from ui.autocomplete import _DELIMITERS as DELIMITERS
from ui.autocomplete import extract_completion_token, replace_completion_token


@pytest.mark.parametrize(
    ("text", "cursor", "expected"),
    [
        ("tag1 AND ha", None, ("ha", 9, 11)),
        ("tag1 AND ha", 10, ("ha", 9, 11)),
        ("tag1, ta", None, ("ta", 6, 8)),
        # 新仕様：() は区切らないので "(" 自体がトークン
        ("(", None, ("(", 0, 1)),
        # 新仕様：":" は区切り。トークンは "ge"
        ("category:ge", None, ("ge", 9, 11)),
    ],
)
def test_extract_completion_token(text: str, cursor: int | None, expected: tuple[str, int, int]) -> None:
    token, start, end = extract_completion_token(text, cursor)
    assert (token, start, end) == expected


@pytest.mark.parametrize(
    ("text", "cursor", "replacement", "expected_text"),
    [
        ("tag1 AND ha", None, "hatsune_miku", "tag1 AND hatsune_miku "),
        # 新仕様：() は区切りではないので、丸ごと "(ta)" が置換対象になる
        ("(ta)", 3, "hatsune_miku", "hatsune_miku "),
        # 新仕様：":" は区切り。category 補完は "ge" を "general" に置換するだけ
        ("category:ge", None, "general", "category:general "),
        ("tag1 AND ta more", 10, "hatsune_miku", "tag1 AND hatsune_miku more"),
    ],
)
def test_replace_completion_token(text: str, cursor: int | None, replacement: str, expected_text: str) -> None:
    _, start, end = extract_completion_token(text, cursor)
    new_text, cursor_pos = replace_completion_token(text, start, end, replacement)
    assert new_text == expected_text

    # カーソル位置の期待値計算は実装と同じロジックで
    suffix = text[end:]
    expected_cursor = start + len(replacement)
    needs_space = (not suffix) or (suffix[0] not in DELIMITERS)
    if replacement and needs_space and not replacement.endswith(" "):
        expected_cursor += 1
    assert cursor_pos == expected_cursor
