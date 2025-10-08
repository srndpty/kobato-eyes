"""Tests for batching behaviour of the directory watcher."""

from __future__ import annotations

from typing import Callable, List, Tuple

import pytest


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
