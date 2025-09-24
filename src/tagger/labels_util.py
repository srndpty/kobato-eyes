"""Helpers for working with WD14 label CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from tagger.base import TagCategory


_DEFAULT_TAG_FILENAMES: tuple[str, ...] = (
    "selected_tags.csv",
    "selected_tags_v3.csv",
    "selected_tags_v3c.csv",
)

_CATEGORY_LOOKUP: dict[str, TagCategory] = {
    "0": TagCategory.GENERAL,
    "general": TagCategory.GENERAL,
    "1": TagCategory.CHARACTER,
    "character": TagCategory.CHARACTER,
    "2": TagCategory.RATING,
    "rating": TagCategory.RATING,
    "3": TagCategory.COPYRIGHT,
    "copyright": TagCategory.COPYRIGHT,
    "4": TagCategory.ARTIST,
    "artist": TagCategory.ARTIST,
    "5": TagCategory.META,
    "meta": TagCategory.META,
}


def _looks_like_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _parse_category(value: str | None) -> TagCategory:
    if not value:
        return TagCategory.GENERAL
    normalised = value.strip().lower()
    if not normalised:
        return TagCategory.GENERAL
    if normalised in _CATEGORY_LOOKUP:
        return _CATEGORY_LOOKUP[normalised]
    if _looks_like_int(normalised):
        try:
            return TagCategory(int(normalised))
        except ValueError:
            return TagCategory.GENERAL
    return TagCategory.GENERAL


def _iter_csv_rows(csv_path: Path) -> Iterable[list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            cells = [cell.strip() for cell in row]
            if not any(cells):
                continue
            if cells[0].startswith("#"):
                continue
            lower_first = cells[0].lower()
            if lower_first in {"tag_id", "tagid", "id", "name", "tag"}:
                continue
            if len(cells) > 1 and cells[1].lower() in {"name", "tag"}:
                continue
            if len(cells) > 2 and cells[2].lower() in {"category"}:
                continue
            yield cells


def load_selected_tags(csv_path: str | Path) -> list[tuple[str, int]]:
    """Parse a WD14 ``selected_tags.csv`` file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file. The file may contain either one, two or four
        columns. Headers and comment lines starting with ``#`` are ignored.

    Returns
    -------
    list[tuple[str, int]]
        Tag names paired with their category as integers.
    """

    path = Path(csv_path)
    labels: list[tuple[str, int]] = []
    for cells in _iter_csv_rows(path):
        name = ""
        category = TagCategory.GENERAL
        if len(cells) == 1:
            name = cells[0]
        elif len(cells) == 2:
            first, second = cells
            if _looks_like_int(first):
                name = second
            else:
                name = first
                category = _parse_category(second)
        else:
            first, second, *rest = cells
            if _looks_like_int(first) and second:
                name = second
                category_value = rest[0] if rest else None
                category = _parse_category(category_value)
            else:
                name = first
                category = _parse_category(second if second else None)
        if name:
            labels.append((name, int(category)))
    return labels


def discover_labels_csv(
    model_path: str | Path | None, tags_csv: str | Path | None
) -> Path | None:
    """Return the path to a WD14 labels CSV if one can be located."""

    if tags_csv:
        candidate = Path(tags_csv)
        return candidate if candidate.exists() else None
    if not model_path:
        return None
    model_file = Path(model_path)
    search_dir = model_file.parent
    candidates: list[Path] = []
    for name in _DEFAULT_TAG_FILENAMES:
        candidate = search_dir / name
        if candidate not in candidates:
            candidates.append(candidate)
    model_candidate = model_file.with_suffix(".csv")
    if model_candidate not in candidates:
        candidates.append(model_candidate)
    for extra in sorted(search_dir.glob("selected_tags*.csv")):
        if extra not in candidates:
            candidates.append(extra)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


__all__ = ["discover_labels_csv", "load_selected_tags"]

