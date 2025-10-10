"""Shared contracts and DTOs for pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class DBItem:
    """Represents the tagging payload for a single file."""

    file_id: int
    tags: Sequence[tuple[str, float, int]]
    width: Optional[int]
    height: Optional[int]
    tagger_sig: Optional[str]
    tagged_at: Optional[float]


@dataclass(frozen=True)
class DBFlush:
    """Explicit flush request for the DB writing queue."""


@dataclass(frozen=True)
class DBStop:
    """Signal message instructing the DB writer to terminate."""


class DBWriteQueue(Protocol):
    """Interface implemented by asynchronous database writers."""

    def start(self) -> None:
        """Start the background worker if necessary."""

    def raise_if_failed(self) -> None:
        """Propagate worker-side exceptions to the caller."""

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        """Enqueue an item for persistence."""

    def qsize(self) -> int:
        """Return the current queue size if supported."""

    def stop(self, *, flush: bool = True, wait_forever: bool = False) -> None:
        """Stop the background worker, optionally flushing pending items."""


__all__ = ["DBItem", "DBFlush", "DBStop", "DBWriteQueue"]
