"""Tests for duplicate status formatting helpers."""

from __future__ import annotations

from types import SimpleNamespace

from ui.dup_status import duplicate_summary, format_duplicate_scan_complete, format_duplicate_summary


def test_duplicate_status_helpers_count_groups_and_files() -> None:
    clusters = [
        SimpleNamespace(files=[object(), object()]),
        SimpleNamespace(files=[object()]),
    ]

    assert duplicate_summary(clusters) == (2, 3)
    assert format_duplicate_summary(clusters) == "2 group(s), 3 file(s) detected."
    assert format_duplicate_scan_complete(clusters) == "Scan complete: 2 group(s), 3 file(s)."
