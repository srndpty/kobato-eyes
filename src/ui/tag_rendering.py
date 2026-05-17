"""Rendering helpers for tag search result labels."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, cast

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from tagger.base import TagCategory
from tagger.categories import build_category_lookup

TagDisplayEntry = tuple[str, float, TagCategory | None]


class _PaletteLike(Protocol):
    """Palette methods used by highlight color selection."""

    def window(self) -> Any:
        """Return a brush-like object with ``color``."""

    def text(self) -> Any:
        """Return a brush-like object with ``color``."""


_CATEGORY_KEY_LOOKUP = build_category_lookup()
_TAG_COLOR_MAP = {
    TagCategory.GENERAL: "#45C5F7",
    TagCategory.CHARACTER: "#63CC69",
    TagCategory.COPYRIGHT: "#C976D8",
}
_SCORE_COLOR = "rgba(255, 255, 255, 0.80)"
_NEUTRAL_TAG_COLOR = "#90A4AE"
_HIGHLIGHT_SCORE_COLOR = "rgba(0, 0, 0, 0.86)"
_TAG_LIST_ROLE = Qt.ItemDataRole.UserRole + 128


def coerce_category(value: object) -> TagCategory | None:
    """Convert a raw category value to :class:`TagCategory` when possible."""

    if value is None:
        return None
    if isinstance(value, TagCategory):
        try:
            return TagCategory(int(value))
        except ValueError:
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return TagCategory(int(value))
        except ValueError:
            return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key:
            return _CATEGORY_KEY_LOOKUP.get(key)
    return None


def filter_tags_by_threshold(tag_rows: Iterable[dict[str, object] | Sequence[object]]) -> list[TagDisplayEntry]:
    """Return displayable tag rows at the fixed copy/export threshold."""

    out: list[TagDisplayEntry] = []
    for row in tag_rows:
        name: object | None = None
        score_value: object | None = None
        category_value: object | None = None

        if isinstance(row, dict):
            name = row.get("name")
            score_value = row.get("score", 0.0)
            category_value = row.get("category")
        else:
            try:
                length = len(row)
            except TypeError:
                continue
            if length == 3:
                name, score_value, category_value = row
            elif length == 2:
                name, score_value = row
            else:
                continue

        try:
            score = float(cast(Any, score_value))
        except (TypeError, ValueError):
            continue

        if score >= 0.1:
            out.append((str(name), float(score), coerce_category(category_value)))

    return out


def relative_luminance(color: QColor) -> float:
    """Return the relative luminance of an sRGB color."""

    def channel(value: int) -> float:
        normalized = value / 255.0
        if normalized <= 0.04045:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(color.red()) + 0.7152 * channel(color.green()) + 0.0722 * channel(color.blue())


def pick_highlight_colors(palette: _PaletteLike) -> tuple[str, str]:
    """Choose highlight background/foreground colors based on the palette."""

    base = palette.window().color()
    text = palette.text().color()
    is_dark = (relative_luminance(base) < 0.5) or (relative_luminance(text) > 0.7)
    if is_dark:
        return "#FFD54F", "#000000"
    return "#FFF59D", "#000000"


def category_color(category: TagCategory | None) -> str:
    """Return the display color for a tag category."""

    if category is None:
        return _NEUTRAL_TAG_COLOR
    return _TAG_COLOR_MAP.get(category, _NEUTRAL_TAG_COLOR)


def mix_hex(fg: str, bg: str, weight: float) -> str:
    """Blend two hex colors."""

    weight = max(0.0, min(1.0, weight))

    def channel_tuple(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    fr, fg_green, fb = channel_tuple(fg)
    br, bg_green, bb = channel_tuple(bg)
    r = round(fr * weight + br * (1 - weight))
    g = round(fg_green * weight + bg_green * (1 - weight))
    b = round(fb * weight + bb * (1 - weight))
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_hex(hex_color: str, factor: float = 0.75) -> str:
    """Darken a hex color by mixing it with black."""

    return mix_hex(hex_color, "#000000", factor)


def render_tag_html(
    name: str,
    score: float,
    category: TagCategory | None,
    *,
    highlight: bool,
    highlight_bg: str,
    highlight_fg: str,
) -> str:
    """Render one tag as HTML for rich item delegates."""

    del highlight_fg
    name_html = html.escape(name)
    score_html = html.escape(f"({score:.2f})")

    if highlight:
        base = category_color(category)
        name_style = f"color:{darken_hex(base, 0.5)};"
        score_style = f"color:{_HIGHLIGHT_SCORE_COLOR};"
        return (
            f'<span style="background-color:{highlight_bg}; padding:0 2px; border-radius:2px;">'
            f'<span style="{name_style}">{name_html}</span> '
            f'<span style="{score_style}">{score_html}</span>'
            f"</span>"
        )

    name_style = f"color:{category_color(category)};"
    score_style = f"color:{_SCORE_COLOR};"
    return f'<span style="{name_style}">{name_html}</span> <span style="{score_style}">{score_html}</span>'


__all__ = [
    "TagDisplayEntry",
    "_SCORE_COLOR",
    "_TAG_LIST_ROLE",
    "coerce_category",
    "filter_tags_by_threshold",
    "pick_highlight_colors",
    "render_tag_html",
]
