"""Pure result-list helpers for the tag search tab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


@dataclass(frozen=True)
class ResultRemovalPlan:
    """Row and paging updates needed after removing result files."""

    rows: list[int]
    offset_removed: int
    next_selection: int


def coerce_file_id(value: object) -> int | None:
    """Return *value* as an integer file id when possible."""

    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def coerce_result_path(value: object) -> Path | None:
    """Return *value* as a non-empty result path when possible."""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return Path(cleaned)


def thumbnail_matches_result(
    results: Sequence[Mapping[str, object]],
    *,
    row: int,
    file_id: int,
) -> bool:
    """Return whether a thumbnail result still belongs to the visible row."""

    if not (0 <= row < len(results)):
        return False
    current_id = coerce_file_id(results[row].get("id"))
    return current_id == int(file_id)


def plan_result_removal(
    results: Sequence[Mapping[str, object]],
    file_ids: Sequence[int],
    *,
    offset_file_ids: Sequence[int],
) -> ResultRemovalPlan | None:
    """Return result-removal bookkeeping, or ``None`` when no visible row matches."""

    id_set = {int(file_id) for file_id in file_ids}
    if not id_set:
        return None
    rows = [
        index
        for index, record in enumerate(results)
        if (file_id := coerce_file_id(record.get("id"))) is not None and file_id in id_set
    ]
    if not rows:
        return None
    offset_id_set = {int(file_id) for file_id in offset_file_ids}
    offset_removed = sum(
        1
        for row in rows
        if (file_id := coerce_file_id(results[row].get("id"))) is not None and file_id in offset_id_set
    )
    next_selection = min(rows[0], max(0, len(results) - len(rows) - 1))
    return ResultRemovalPlan(rows=rows, offset_removed=offset_removed, next_selection=next_selection)


def should_queue_missing_thumbnail(
    record: Mapping[str, object],
    *,
    has_thumbnail: bool,
) -> tuple[int, Path] | None:
    """Return thumbnail work for a row that lacks a current decoration."""

    if has_thumbnail:
        return None
    file_id = coerce_file_id(record.get("id"))
    path = coerce_result_path(record.get("path"))
    if file_id is None or path is None or not path.exists():
        return None
    return file_id, path


__all__ = [
    "ResultRemovalPlan",
    "coerce_file_id",
    "coerce_result_path",
    "plan_result_removal",
    "should_queue_missing_thumbnail",
    "thumbnail_matches_result",
]
