"""Helpers for interrogating runtime environment flags."""

from __future__ import annotations

import os


def is_headless() -> bool:
    """Return True when the application should avoid Qt GUI features."""
    value = os.environ.get("KOE_HEADLESS", "")
    return value.lower() not in {"", "0", "false", "no"}
