"""Tests for tag tab control-state helpers."""

from __future__ import annotations

from ui.tags_control_state import TagsActivityState, compute_tags_control_availability


def test_tags_controls_are_enabled_when_idle_with_query() -> None:
    availability = compute_tags_control_availability(
        TagsActivityState(
            indexing_active=False,
            search_busy=False,
            refresh_active=False,
            has_current_query=True,
            can_load_more=True,
        )
    )

    assert availability.search is True
    assert availability.query_input is True
    assert availability.load_more is True
    assert availability.retag is True
    assert availability.retag_results is True
    assert availability.refresh is True
    assert availability.copy_results is True


def test_tags_controls_disable_mutating_actions_while_indexing() -> None:
    availability = compute_tags_control_availability(
        TagsActivityState(
            indexing_active=True,
            search_busy=False,
            refresh_active=False,
            has_current_query=True,
            can_load_more=True,
        )
    )

    assert availability.search is False
    assert availability.query_input is False
    assert availability.load_more is False
    assert availability.placeholder is False
    assert availability.table_view is False
    assert availability.grid_view is False
    assert availability.retag is False
    assert availability.retag_results is False
    assert availability.refresh is False
    assert availability.copy_results is False


def test_tags_controls_disable_result_actions_while_searching_or_refreshing() -> None:
    searching = compute_tags_control_availability(
        TagsActivityState(
            indexing_active=False,
            search_busy=True,
            refresh_active=False,
            has_current_query=True,
            can_load_more=True,
        )
    )
    refreshing = compute_tags_control_availability(
        TagsActivityState(
            indexing_active=False,
            search_busy=False,
            refresh_active=True,
            has_current_query=True,
            can_load_more=True,
        )
    )

    assert searching.search is True
    assert searching.load_more is False
    assert searching.retag is False
    assert searching.refresh is False
    assert searching.copy_results is False
    assert refreshing.refresh is False
    assert refreshing.copy_results is False
