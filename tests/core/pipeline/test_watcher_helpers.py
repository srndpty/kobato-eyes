"""Tests for watcher path scheduling helpers."""

from __future__ import annotations

from pathlib import Path

from core.pipeline.watcher import resolve_watch_paths


def test_resolve_watch_paths_filters_duplicates_and_extensions(tmp_path: Path) -> None:
    first = tmp_path / "a.PNG"
    duplicate = tmp_path / "a.PNG"
    ignored = tmp_path / "note.txt"

    resolved, scheduled = resolve_watch_paths([first, duplicate, ignored], allow_exts={".png"})

    assert resolved == [first.expanduser().resolve()]
    assert scheduled == {first.expanduser().resolve()}


def test_resolve_watch_paths_skips_already_scheduled(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    existing = first.expanduser().resolve()

    resolved, scheduled = resolve_watch_paths([first], allow_exts={".png"}, already_scheduled={existing})

    assert resolved == []
    assert scheduled == set()
