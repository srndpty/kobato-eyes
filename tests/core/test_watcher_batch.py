"""Tests for batching behaviour of the directory watcher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Tuple

import pytest

from core.watcher import DirectoryWatcher


class _DummyObserver:
    """Minimal watchdog observer stub used for unit tests."""

    def __init__(self) -> None:
        self.handler = None
        self.args = None
        self.started = False

    def schedule(self, handler, path: str, recursive: bool) -> None:  # noqa: D401
        self.handler = handler
        self.args = (path, recursive)

    def start(self) -> None:  # noqa: D401
        self.started = True

    def stop(self) -> None:  # noqa: D401
        self.started = False

    def join(self, timeout: float | None = None) -> None:  # noqa: D401
        return None


@pytest.fixture()
def observer_factory() -> Tuple[Callable[[], _DummyObserver], List[_DummyObserver]]:
    created: List[_DummyObserver] = []

    def factory() -> _DummyObserver:
        observer = _DummyObserver()
        created.append(observer)
        return observer

    return factory, created


def test_batching_and_deduplication(
    tmp_path: Path, observer_factory: Tuple[Callable[[], _DummyObserver], List[_DummyObserver]]
) -> None:
    factory, created = observer_factory
    calls: list[list[Path]] = []

    watcher = DirectoryWatcher(
        roots=[tmp_path],
        callback=lambda paths: calls.append(paths),
        extensions={".png"},
        use_qtimer=False,
        observer_factory=factory,
    )
    watcher.start()

    assert created
    handler = created[0].handler
    assert handler is not None

    first_image = tmp_path / "first.png"
    first_image.write_bytes(b"data")

    event = SimpleNamespace(src_path=str(first_image), is_directory=False)
    handler.on_created(event)
    handler.on_modified(event)

    assert not calls

    watcher.flush_pending()
    assert len(calls) == 1
    assert calls[0] == [first_image.resolve()]

    calls.clear()

    moved_target = tmp_path / "second.png"
    moved_target.write_bytes(b"more")
    moved_event = SimpleNamespace(src_path=str(first_image), dest_path=str(moved_target), is_directory=False)
    handler.on_moved(moved_event)

    watcher.flush_pending()
    assert len(calls) == 1
    assert calls[0] == [moved_target.resolve()]

    watcher.stop()
