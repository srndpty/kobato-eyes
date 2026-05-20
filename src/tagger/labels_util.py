"""Helpers for working with WD14 label CSV files."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

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
    ips: tuple[str, ...] = ()


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


_HEADER_ALIASES: dict[str, str] = {
    "id": "row_id",
    "tag_id": "tag_id",
    "tagid": "tag_id",
    "tag": "name",
    "name": "name",
    "category": "category",
    "count": "count",
    "ips": "ips",
}


def _header_columns(cells: list[str]) -> dict[str, int] | None:
    columns: dict[str, int] = {}
    for index, cell in enumerate(cells):
        key = _HEADER_ALIASES.get(cell.strip().lower())
        if key is None:
            continue
        columns.setdefault(key, index)
    if "name" not in columns:
        return None
    if len(columns) < 2:
        return None
    return columns


def _parse_ips(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    candidate = value.strip()
    if not candidate:
        return ()
    try:
        parsed = json.loads(candidate)
    except Exception:
        return ()
    if isinstance(parsed, str):
        cleaned = parsed.strip()
        return (cleaned,) if cleaned else ()
    if isinstance(parsed, (list, tuple)):
        ips: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                ips.append(cleaned)
        return tuple(ips)
    return ()


BROKEN_TAG_PREFIX = "__pixai_broken_"


def _parse_row(cells: list[str], columns: dict[str, int] | None = None) -> TagMeta | None:
    if columns is not None:
        return _parse_headered_row(cells, columns)

    name = ""
    category = 0
    count = 0
    ips: tuple[str, ...] = ()
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
        if cell_count > 1 and _looks_like_int(cells[1]):
            # Legacy/export format: id,tag_id,name,category,count,ips
            name = cells[2] if cell_count > 2 else ""
            category = _parse_category(cells[3] if cell_count > 3 else None)
            count = _parse_count(cells[4] if cell_count > 4 else None)
            if cell_count > 5:
                ips = _parse_ips(cells[5])
        else:
            # Common WD14 format: tag_id,name,category,count[,ips]
            name = cells[1] if cell_count > 1 else ""
            category = _parse_category(cells[2] if cell_count > 2 else None)
            count = _parse_count(cells[3] if cell_count > 3 else None)
            if cell_count > 4:
                ips = _parse_ips(cells[4])
    else:
        first = cells[0]
        name = first
        third = cells[2] if cell_count > 2 else ""
        category = _parse_category(cells[1] if cell_count > 1 else None)
        if cell_count > 2:
            count = _parse_count(third)
        if cell_count > 3:
            ips = _parse_ips(cells[3])
    cleaned = name.strip()
    if not cleaned:
        # 空nameの行は捨てずにプレースホルダを合成して返す（次元を維持）
        rid: int | None = None
        if _looks_like_int(cells[0]):
            rid = int(cells[0])
        elif len(cells) > 1 and _looks_like_int(cells[1]):
            rid = int(cells[1])
        placeholder = f"{BROKEN_TAG_PREFIX}{rid if rid is not None else 'unknown'}"
        # カテゴリはMETA(=5) に寄せておくと閾値の影響を受けにくい
        return TagMeta(name=placeholder, category=5, count=0, ips=())
    return TagMeta(name=cleaned, category=category, count=count, ips=ips)


def _cell_at(cells: list[str], index: int | None) -> str | None:
    if index is None:
        return None
    if index >= len(cells):
        return None
    return cells[index]


def _parse_headered_row(cells: list[str], columns: dict[str, int]) -> TagMeta | None:
    name = _cell_at(cells, columns.get("name")) or ""
    category = _parse_category(_cell_at(cells, columns.get("category")))
    count = _parse_count(_cell_at(cells, columns.get("count")))
    ips = _parse_ips(_cell_at(cells, columns.get("ips")))
    cleaned = name.strip()
    if cleaned:
        return TagMeta(name=cleaned, category=category, count=count, ips=ips)

    rid: int | None = None
    for key in ("tag_id", "row_id"):
        value = _cell_at(cells, columns.get(key))
        if value is not None and _looks_like_int(value):
            rid = int(value)
            break
    placeholder = f"{BROKEN_TAG_PREFIX}{rid if rid is not None else 'unknown'}"
    return TagMeta(name=placeholder, category=5, count=0, ips=())


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
    missing: list[tuple[int, list[str]]] = []
    columns: dict[str, int] | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for lineno, row in enumerate(reader, start=1):
            if not row:
                continue
            cells = [cell.strip() for cell in row]
            if not any(cells):
                continue
            if cells[0].startswith("#"):
                continue
            header = _header_columns(cells)
            if header is not None:
                columns = header
                continue

            tag = _parse_row(cells, columns)
            if tag is not None:
                labels.append(tag)
            else:
                missing.append((lineno, cells))
    if missing:
        import logging

        logger = logging.getLogger(__name__)
        for lineno, cells in missing[:5]:  # 多すぎると邪魔なので先頭5件だけ
            logger.error("CSV row dropped (lineno=%d): %r", lineno, cells)
        if len(missing) > 5:
            logger.error("...and %d more dropped rows", len(missing) - 5)
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
