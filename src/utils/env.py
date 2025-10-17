"""Helpers for interrogating runtime environment flags."""

from __future__ import annotations

import os
from typing import SupportsIndex, SupportsInt


def is_headless() -> bool:
    """Return True when the application should avoid Qt GUI features."""
    value = os.environ.get("KOE_HEADLESS", "")
    return value.lower() not in {"", "0", "false", "no"}


def safe_int(
    value: SupportsInt | SupportsIndex | str | None,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Safely coerce ``value`` to :class:`int`, returning ``default`` on failure.

    The function mirrors the behaviour expected when reading configuration
    values from environment variables. Empty strings, ``None`` and malformed
    values silently fall back to ``default``. Optional ``min_value`` and
    ``max_value`` bounds can be supplied to clamp the acceptable range; any
    value outside the range is treated as invalid and therefore falls back to
    ``default``.
    """

    candidate = value
    if candidate is None:
        return default
    if isinstance(candidate, str):
        candidate = candidate.strip()
        if candidate == "":
            return default

    try:
        coerced = int(candidate)
    except (TypeError, ValueError):
        return default

    if min_value is not None and coerced < min_value:
        return default
    if max_value is not None and coerced > max_value:
        return default
    return coerced


__all__ = ["is_headless", "safe_int"]
