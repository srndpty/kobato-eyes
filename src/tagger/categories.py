"""Helpers for working with :class:`tagger.base.TagCategory`."""

from __future__ import annotations

from .base import TagCategory


def build_category_lookup(*, include_numeric: bool = True) -> dict[str, TagCategory]:
    """Return a lookup table mapping string keys to :class:`TagCategory` values.

    Parameters
    ----------
    include_numeric:
        When ``True`` the lookup will include the decimal string form of the
        category value (e.g. ``"0"`` for :class:`TagCategory.GENERAL`).
    """

    mapping: dict[str, TagCategory] = {}
    for category in TagCategory:
        mapping[category.name.lower()] = category
        if include_numeric:
            mapping[str(int(category))] = category
    return mapping


__all__ = ["build_category_lookup"]
