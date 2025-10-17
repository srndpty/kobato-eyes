"""Filesystem helpers shared across kobato-eyes modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

WINDOWS = os.name == "nt"
LONG_PATH_PREFIX = "\\\\?\\"


def to_system_path(path: Path) -> str:
    """Return a string suitable for low-level filesystem APIs (handles long paths)."""
    abs_path = Path(path).resolve()
    path_str = str(abs_path)
    if not WINDOWS:
        return path_str

    if path_str.startswith(LONG_PATH_PREFIX):
        return path_str

    if len(path_str) >= 248:
        return f"{LONG_PATH_PREFIX}{path_str}"

    return path_str


def from_system_path(path: str) -> Path:
    """Convert a system path string back into a Path object."""
    if WINDOWS and path.startswith(LONG_PATH_PREFIX):
        return Path(path[len(LONG_PATH_PREFIX) :])
    return Path(path)


def is_hidden(path: Path) -> bool:
    """Identify hidden or system paths across platforms."""
    candidate = Path(path)
    name = candidate.name

    if name.startswith(".") and name not in (".", ".."):
        return True

    if not WINDOWS:
        return False

    try:
        import ctypes
    except Exception:  # pragma: no cover - defensive fallback
        return False

    windll = getattr(ctypes, "windll", None)
    kernel32 = getattr(windll, "kernel32", None)
    get_attrs = getattr(kernel32, "GetFileAttributesW", None)
    if not callable(get_attrs):
        return False

    try:
        attrs = int(get_attrs(str(candidate)))
    except Exception:  # pragma: no cover - defensive fallback
        return False

    if attrs == -1:
        return False

    file_attribute_hidden = 0x2
    file_attribute_system = 0x4
    return bool(attrs & (file_attribute_hidden | file_attribute_system))


def path_in_roots(path: Path, roots: Iterable[Path]) -> bool:
    """Check whether a path is located under any of the provided root directories."""
    candidate = Path(path).resolve()
    for root in roots:
        root_resolved = Path(root).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            continue
        return True
    return False


__all__ = [
    "WINDOWS",
    "LONG_PATH_PREFIX",
    "to_system_path",
    "from_system_path",
    "is_hidden",
    "path_in_roots",
]
