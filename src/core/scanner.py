"""Filesystem scanning utilities for kobato-eyes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from utils.fs import from_system_path, is_hidden, to_system_path

DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
}


def _normalise_extensions(extensions: Iterable[str] | None) -> set[str] | None:
    if extensions is None:
        return None
    normalised: set[str] = set()
    for ext in extensions:
        candidate = ext.lower()
        if not candidate.startswith("."):
            candidate = f".{candidate}"
        normalised.add(candidate)
    return normalised


def _is_excluded(path: Path, excluded_paths: Sequence[Path]) -> bool:
    for excluded in excluded_paths:
        try:
            Path(path).resolve().relative_to(Path(excluded).resolve())
            return True
        except ValueError:
            continue
    return False


def iter_images(
    roots: Iterable[str | Path],
    *,
    excluded: Iterable[str | Path] | None = None,
    extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield image files located beneath `roots`, applying filters for exclusions."""
    root_paths = [Path(root).resolve() for root in roots]
    excluded_paths = [Path(path).resolve() for path in (excluded or [])]
    allowed_extensions = _normalise_extensions(extensions) or DEFAULT_EXTENSIONS

    for root_path in root_paths:
        if not root_path.exists() or not root_path.is_dir():
            continue

        system_root = to_system_path(root_path)
        for dirpath, dirnames, filenames in os.walk(system_root):
            current_dir = from_system_path(dirpath)

            if is_hidden(current_dir) or _is_excluded(current_dir, excluded_paths):
                dirnames[:] = []
                continue

            dirnames[:] = [
                name
                for name in dirnames
                if not is_hidden(current_dir / name) and not _is_excluded(current_dir / name, excluded_paths)
            ]

            for filename in filenames:
                file_path = current_dir / filename
                if is_hidden(file_path) or _is_excluded(file_path, excluded_paths):
                    continue
                if allowed_extensions and file_path.suffix.lower() not in allowed_extensions:
                    continue
                yield file_path


__all__ = ["iter_images"]
