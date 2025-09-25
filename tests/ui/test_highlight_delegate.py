import pytest

try:
    from ui.tags_tab import _HighlightDelegate
except ImportError as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(f"PyQt6 not available: {exc}", allow_module_level=True)


HIGHLIGHT_START = '<span style="background-color:#fff3a3;">'
HIGHLIGHT_END = '</span>'


@pytest.mark.not_gui
def test_highlight_multiple_occurrences() -> None:
    result = _HighlightDelegate._to_html_with_highlight("Nurse nurse", ["nurse"])
    assert (
        result
        == f"{HIGHLIGHT_START}Nurse{HIGHLIGHT_END} {HIGHLIGHT_START}nurse{HIGHLIGHT_END}"
    )


@pytest.mark.not_gui
def test_highlight_prefers_longer_span() -> None:
    result = _HighlightDelegate._to_html_with_highlight("nurse_cap", ["nurse_cap", "nurse"])
    assert result == f"{HIGHLIGHT_START}nurse_cap{HIGHLIGHT_END}"


@pytest.mark.not_gui
def test_highlight_escapes_html() -> None:
    result = _HighlightDelegate._to_html_with_highlight("<nurse & co>", ["nurse"])
    assert result == f"&lt;{HIGHLIGHT_START}nurse{HIGHLIGHT_END} &amp; co&gt;"
