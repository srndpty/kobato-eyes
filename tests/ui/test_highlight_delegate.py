# tests/ui/test_highlight_delegate.py

from __future__ import annotations

import re
from html import escape

import pytest

from ui.tags_tab import _HighlightDelegate

BACKGROUND = "#ffee77"
FOREGROUND = "#000000"


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _highlight_blocks(html: str) -> list[str]:
    """背景色付きハイライトのブロック（外側のspanの中身）を抜き出す。"""
    pat = re.compile(
        r"<span[^>]*background-color:\s*" + re.escape(BACKGROUND) + r"[^>]*>(.*?)</span>",
        re.IGNORECASE | re.DOTALL,
    )
    return [m.group(1) for m in pat.finditer(html)]


def _has_highlight_for_term(html: str, term: str) -> bool:
    """指定語がハイライトブロック内に含まれるか（タグを跨いでもOK）。"""
    esc = re.escape(escape(term))
    for block in _highlight_blocks(html):
        if re.search(esc, block, re.IGNORECASE | re.DOTALL):
            return True
    return False


@pytest.mark.not_gui
def test_highlight_exact_match_only_with_tags() -> None:
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["nurse"],
        tags=[("nurse", 0.91), ("nurse_cap", 0.80)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )

    # 素のテキストとして正しい並び
    assert _strip_tags(html) == "nurse (0.91), nurse_cap (0.80)"

    # nurse だけがハイライト
    assert _has_highlight_for_term(html, "nurse")
    assert not _has_highlight_for_term(html, "nurse_cap")


@pytest.mark.not_gui
def test_highlight_is_case_insensitive() -> None:
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["NuRSe"],  # 大文字小文字は無視
        tags=[("nurse", 0.75), ("doctor", 0.50)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )

    assert _strip_tags(html) == "nurse (0.75), doctor (0.50)"
    assert _has_highlight_for_term(html, "nurse")
    assert not _has_highlight_for_term(html, "doctor")


@pytest.mark.not_gui
def test_highlight_escapes_html_in_names() -> None:
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["<nurse & co>"],
        tags=[("<nurse & co>", 0.50), ("doctor", 0.50)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )

    # エスケープ後の素テキストで比較
    assert _strip_tags(html) == f"{escape('<nurse & co>')} (0.50), doctor (0.50)"
    assert _has_highlight_for_term(html, "<nurse & co>")
    assert not _has_highlight_for_term(html, "doctor")


@pytest.mark.not_gui
def test_highlight_multiple_terms() -> None:
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["nurse", "doctor"],
        tags=[("patient", 0.33), ("doctor", 0.80), ("nurse", 0.70)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )

    assert _strip_tags(html) == "patient (0.33), doctor (0.80), nurse (0.70)"
    # doctor / nurse がハイライト、patient は非ハイライト
    assert _has_highlight_for_term(html, "doctor")
    assert _has_highlight_for_term(html, "nurse")
    assert not _has_highlight_for_term(html, "patient")

    # ハイライトの総数も 2 のはず
    assert len(_highlight_blocks(html)) == 2


@pytest.mark.not_gui
def test_highlight_no_match_results_in_plain_list() -> None:
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["surgeon"],
        tags=[("nurse", 0.40), ("doctor", 0.60)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )

    # 素のテキスト比較 + ハイライトが無いこと
    assert _strip_tags(html) == "nurse (0.40), doctor (0.60)"
    assert len(_highlight_blocks(html)) == 0
