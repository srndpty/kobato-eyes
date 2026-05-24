"""Pure action helpers for tag search results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from ui.tag_rendering import filter_tags_by_threshold
from ui.tags_result_state import coerce_file_id, coerce_result_path


@dataclass(frozen=True)
class CopyTagsPayload:
    """Clipboard text and user feedback for a copy-tags action."""

    text: str
    feedback: str


def result_row_from_stored(
    index_row: int,
    stored_row: object,
    *,
    result_count: int,
) -> int | None:
    """Resolve a model index row or stored result row into a valid result row."""

    try:
        row = int(cast(Any, stored_row)) if stored_row is not None else int(index_row)
    except (TypeError, ValueError):
        return None
    if 0 <= row < result_count:
        return row
    return None


def normalize_selected_rows(rows: Iterable[object], *, result_count: int) -> list[int]:
    """Return sorted unique valid result rows."""

    normalized: set[int] = set()
    for value in rows:
        try:
            row = int(cast(Any, value))
        except (TypeError, ValueError):
            continue
        if 0 <= row < result_count:
            normalized.add(row)
    return sorted(normalized)


def collect_delete_entries(
    results: Sequence[Mapping[str, object]],
    rows: Sequence[int],
) -> list[tuple[int, Path]]:
    """Return database ids and paths for selected result rows that can be deleted."""

    entries: list[tuple[int, Path]] = []
    for row in rows:
        if not (0 <= row < len(results)):
            continue
        record = results[row]
        file_id = coerce_file_id(record.get("id"))
        path = coerce_result_path(record.get("path"))
        if file_id is None or path is None:
            continue
        entries.append((file_id, path))
    return entries


def build_copy_tags_payload(raw_tags: Iterable[Sequence[object]], *, include_scores: bool) -> CopyTagsPayload | None:
    """Return clipboard text and feedback for filtered result tags."""

    filtered = filter_tags_by_threshold(raw_tags)
    if not filtered:
        return None
    if include_scores:
        text = ", ".join(f"{name} ({score:.2f})" for name, score, _ in filtered)
        feedback = "タグ（スコア付き）をクリップボードにコピーしました。"
    else:
        text = ", ".join(name for name, _, _ in filtered)
        feedback = "タグをクリップボードにコピーしました。"
    return CopyTagsPayload(text=text, feedback=feedback)


__all__ = [
    "CopyTagsPayload",
    "build_copy_tags_payload",
    "collect_delete_entries",
    "normalize_selected_rows",
    "result_row_from_stored",
]
