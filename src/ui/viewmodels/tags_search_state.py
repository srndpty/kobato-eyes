"""State container for asynchronous tag searches."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TagsSearchState:
    """Track mutable state for one tag-search UI session."""

    offset: int = 0
    busy: bool = False
    can_load_more: bool = False
    generation: int = 0
    reset_pending: bool = False
    received_any: bool = False
    last_cancelled: bool = False
    generations_reset: dict[int, bool] = field(default_factory=dict)

    def begin_query(self) -> None:
        """Reset counters for a new user-submitted query."""

        self.offset = 0
        self.reset_pending = True
        self.received_any = False
        self.last_cancelled = False
        self.can_load_more = False

    def begin_worker(self, *, reset: bool) -> int:
        """Register a newly started search worker and return its generation."""

        self.generation += 1
        self.generations_reset[self.generation] = bool(reset)
        self.received_any = False
        return self.generation

    def consume_rows(self, row_count: int, *, chunk_size: int) -> None:
        """Update paging state after a chunk arrives."""

        if row_count <= 0:
            return
        self.received_any = True
        self.offset += int(row_count)
        self.can_load_more = int(row_count) == int(chunk_size)

    def finish_generation(self, generation: int) -> bool:
        """Return whether the finished generation represented a reset search."""

        return self.generations_reset.pop(generation, False)

    def discard_generation(self, generation: int) -> None:
        """Forget bookkeeping for an obsolete or failed generation."""

        self.generations_reset.pop(generation, None)


__all__ = ["TagsSearchState"]
