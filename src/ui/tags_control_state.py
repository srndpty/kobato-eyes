"""Control-state helpers for the tag search tab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagsActivityState:
    """Runtime activity flags that affect tag tab controls."""

    indexing_active: bool
    search_busy: bool
    refresh_active: bool
    has_current_query: bool
    can_load_more: bool
    delete_active: bool = False


@dataclass(frozen=True)
class TagsControlAvailability:
    """Computed enabled states for tag tab controls."""

    search: bool
    query_input: bool
    load_more: bool
    placeholder: bool
    table_view: bool
    grid_view: bool
    retag: bool
    retag_results: bool
    refresh: bool
    copy_results: bool


def compute_tags_control_availability(state: TagsActivityState) -> TagsControlAvailability:
    """Return enabled states for tag tab controls from activity flags."""

    idle_for_index = not state.indexing_active and not state.delete_active
    idle_for_search_action = not state.indexing_active and not state.search_busy and not state.delete_active
    refresh = not state.refresh_active and idle_for_search_action
    copy_results = state.has_current_query and not (
        state.search_busy or state.indexing_active or state.refresh_active or state.delete_active
    )
    return TagsControlAvailability(
        search=idle_for_index,
        query_input=idle_for_index,
        load_more=state.can_load_more and idle_for_search_action,
        placeholder=idle_for_index,
        table_view=idle_for_index,
        grid_view=idle_for_index,
        retag=idle_for_search_action,
        retag_results=state.has_current_query and idle_for_search_action,
        refresh=refresh,
        copy_results=copy_results,
    )


__all__ = ["TagsActivityState", "TagsControlAvailability", "compute_tags_control_availability"]
