"""Helpers for working with WD14 label CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

_DEFAULT_TAG_FILENAMES: tuple[str, ...] = (
    "selected_tags.csv",
    "selected_tags_v3.csv",
    "selected_tags_v3c.csv",
)

_CATEGORY_LOOKUP: dict[str, int] = {
    "0": 0,
    "general": 0,
    "1": 1,
    "character": 1,
    "2": 2,
    "rating": 2,
    "3": 3,
    "copyright": 3,
    "4": 4,
    "artist": 4,
    "5": 5,
    "meta": 5,
}


@dataclass(frozen=True)
class TagMeta:
    """Basic metadata describing a single tag."""

    name: str
    category: int
    count: int | None = None


def _looks_like_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _parse_category(value: str | None) -> int:
    if not value:
        return 0
    normalised = value.strip().lower()
    if not normalised:
        return 0
    if normalised in _CATEGORY_LOOKUP:
        return _CATEGORY_LOOKUP[normalised]
    if _looks_like_int(normalised):
        try:
            return int(normalised)
        except ValueError:
            return 0
    return 0


def _parse_count(value: str | None) -> int:
    if not value:
        return 0
    stripped = value.strip()
    if not stripped:
        return 0
    try:
        return int(float(stripped))
    except ValueError:
        return 0


def _iter_csv_rows(csv_path: Path) -> Iterator[list[str]]:
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
            # tagという名前のタグがあるので、ここでtagで除外してはいけない
            if len(cells) > 1 and cells[1].lower() in {"name"}:
                continue
            if len(cells) > 2 and cells[2].lower() in {"category"}:
                continue
            yield cells


def _parse_row(cells: list[str]) -> TagMeta | None:
    name = ""
    category = 0
    count = 0
    cell_count = len(cells)
    if cell_count == 1:
        name = cells[0]
    elif cell_count == 2:
        first, second = cells
        if _looks_like_int(first):
            name = second
        else:
            name = first
            category = _parse_category(second)
    elif cell_count >= 3 and _looks_like_int(cells[0]):
        padded = (cells + ["", "", "", ""])[:4]
        name = padded[1]
        category = _parse_category(padded[2])
        count = _parse_count(padded[3])
    else:
        first = cells[0]
        name = first
        second = cells[1] if cell_count > 1 else None
        third = cells[2] if cell_count > 2 else None
        category = _parse_category(second)
        if cell_count > 2:
            count = _parse_count(third)
    cleaned = name.strip()
    if not cleaned:
        return None
    return TagMeta(name=cleaned, category=category, count=count)


def load_selected_tags(csv_path: str | Path) -> list[TagMeta]:
    """Parse a WD14 ``selected_tags.csv`` file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file. The file may contain either one, two or four
        columns. Headers and comment lines starting with ``#`` are ignored.

    Returns
    -------
    list[TagMeta]
        Metadata for each tag.
    """

    path = Path(csv_path)
    labels: list[TagMeta] = []
    for cells in _iter_csv_rows(path):
        tag = _parse_row(cells)
        if tag is not None:
            labels.append(tag)
    return labels


def discover_labels_csv(model_path: str | Path | None, tags_csv: str | Path | None) -> Path | None:
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


def sort_by_popularity(tags: Iterable[TagMeta]) -> list[TagMeta]:
    """Return tags ordered by count (descending) then name (ascending)."""

    return sorted(tags, key=lambda tag: (-int(tag.count or 0), tag.name.lower()))


__all__ = ["TagMeta", "discover_labels_csv", "load_selected_tags", "sort_by_popularity"]
