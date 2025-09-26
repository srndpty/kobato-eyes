# tests/ui/test_highlight_delegate.py

from __future__ import annotations

import pytest

from ui.tags_tab import _HighlightDelegate

BACKGROUND = "#ffee77"
FOREGROUND = "#000000"


@pytest.mark.not_gui
def test_highlight_exact_match_only_with_tags() -> None:
    # terms に含まれる "nurse" と完全一致の tag だけを強調
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["nurse"],
        tags=[("nurse", 0.91), ("nurse_cap", 0.80)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )
    assert html == (
        f'<span style="background-color:{BACKGROUND}; color:{FOREGROUND};">nurse (0.91)</span>, ' "nurse_cap (0.80)"
    )


@pytest.mark.not_gui
def test_highlight_is_case_insensitive() -> None:
    # 大文字小文字は無視して一致
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["NuRSe"],
        tags=[("nurse", 0.75), ("doctor", 0.50)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )
    assert html == (
        f'<span style="background-color:{BACKGROUND}; color:{FOREGROUND};">nurse (0.75)</span>, ' "doctor (0.50)"
    )


@pytest.mark.not_gui
def test_highlight_escapes_html_in_names() -> None:
    # タグ名中の <>& などは必ずエスケープされる
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["<nurse & co>"],
        tags=[("<nurse & co>", 0.50), ("doctor", 0.50)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )
    assert html == (
        f'<span style="background-color:{BACKGROUND}; color:{FOREGROUND};">'
        "&lt;nurse &amp; co&gt; (0.50)</span>, doctor (0.50)"
    )


@pytest.mark.not_gui
def test_highlight_multiple_terms() -> None:
    # 複数語。合致するものだけそれぞれ強調し、合致しないものは素のまま
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["nurse", "doctor"],
        tags=[("patient", 0.33), ("doctor", 0.80), ("nurse", 0.70)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )
    assert html == (
        "patient (0.33), "
        f'<span style="background-color:{BACKGROUND}; color:{FOREGROUND};">doctor (0.80)</span>, '
        f'<span style="background-color:{BACKGROUND}; color:{FOREGROUND};">nurse (0.70)</span>'
    )


@pytest.mark.not_gui
def test_highlight_no_match_results_in_plain_list() -> None:
    # terms にマッチが無ければ強調なしでそのまま出力
    html = _HighlightDelegate._to_html_with_highlight(
        text="ignored",
        terms=["surgeon"],
        tags=[("nurse", 0.40), ("doctor", 0.60)],
        bg=BACKGROUND,
        fg=FOREGROUND,
    )
    assert html == "nurse (0.40), doctor (0.60)"
