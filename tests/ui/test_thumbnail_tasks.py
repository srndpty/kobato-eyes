"""Tests for thumbnail worker task boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)

from ui import thumbnail_tasks


class _Emitter:
    def __init__(self) -> None:
        self.emitted: list[tuple[int, object]] = []

    def emit(self, row: int, pixmap: object) -> None:
        """Collect emitted thumbnail payloads."""

        self.emitted.append((row, pixmap))


class _Signal:
    def __init__(self) -> None:
        self.finished = _Emitter()


def test_thumbnail_task_loads_thumbnail_and_emits_row(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    pixmap = object()
    signal = _Signal()
    calls: list[tuple[Path, int, int]] = []

    def fake_get_thumbnail(image_path: Path, width: int, height: int) -> object:
        calls.append((image_path, width, height))
        return pixmap

    monkeypatch.setattr(thumbnail_tasks, "get_thumbnail", fake_get_thumbnail)

    task = thumbnail_tasks.ThumbnailTask(3, path, 128, 96, signal)  # type: ignore[arg-type]
    task.run()

    assert calls == [(path, 128, 96)]
    assert signal.finished.emitted == [(3, pixmap)]


def test_thumbnail_task_cancel_skips_load_and_emit(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    signal = _Signal()
    calls: list[Path] = []

    def fake_get_thumbnail(image_path: Path, width: int, height: int) -> object:
        calls.append(image_path)
        return object()

    monkeypatch.setattr(thumbnail_tasks, "get_thumbnail", fake_get_thumbnail)

    task = thumbnail_tasks.ThumbnailTask(3, path, 128, 96, signal)  # type: ignore[arg-type]
    task.cancel()
    task.run()

    assert calls == []
    assert signal.finished.emitted == []


def test_thumbnail_task_broken_image_emits_empty_pixmap(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    signal = _Signal()
    empty_pixmap = object()

    def fake_get_thumbnail(image_path: Path, width: int, height: int) -> object:
        raise OSError("broken image")

    monkeypatch.setattr(thumbnail_tasks, "get_thumbnail", fake_get_thumbnail)
    monkeypatch.setattr(thumbnail_tasks, "QPixmap", lambda: empty_pixmap)

    task = thumbnail_tasks.ThumbnailTask(4, path, 64, 64, signal)  # type: ignore[arg-type]
    task.run()

    assert signal.finished.emitted == [(4, empty_pixmap)]
