"""Filesystem scanning utilities for kobato-eyes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

from utils.fs import is_hidden

DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
}

def _normalise_paths(paths: Iterable[str | Path] | None) -> list[Path]:
    """Expand and safely resolve a sequence of filesystem paths."""

    normalised: list[Path] = []
    if not paths:
        return normalised

    for item in paths:
        if item is None:
            continue
        candidate = Path(item).expanduser()
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate.absolute()
        normalised.append(resolved)
    return normalised


def _has_hidden_component(path: Path) -> bool:
    """Determine whether any component in the path is hidden."""

    current = Path(path)
    while True:
        if is_hidden(current):
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _path_in(path: Path, bases: Sequence[Path]) -> bool:
    """Check whether *path* is located under any path in *bases*."""

    candidate = Path(path)
    for base in bases:
        base_path = Path(base)
        if hasattr(candidate, "is_relative_to"):
            if candidate.is_relative_to(base_path):  # type: ignore[attr-defined]
                return True
            continue
        base_parts = base_path.parts
        candidate_parts = candidate.parts
        if len(candidate_parts) < len(base_parts):
            continue
        if candidate_parts[: len(base_parts)] == base_parts:
            return True
    return False


def iter_images(
    roots: Iterable[str | Path],
    *,
    excluded: Iterable[str | Path] | None = None,
    extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield image files located beneath `roots`, applying filters for exclusions."""
    exts = {
        (ext.lower() if ext.startswith(".") else f".{ext.lower()}")
        for ext in (extensions or DEFAULT_EXTENSIONS)
    }
    excluded_paths = _normalise_paths(excluded)

    for root in roots:
        root_path = Path(root).expanduser()
        try:
            resolved_root = root_path.resolve(strict=False)
        except OSError:
            resolved_root = root_path.absolute()

        if not resolved_root.exists() or not resolved_root.is_dir():
            continue

        for path in resolved_root.rglob("*"):
            if not path.is_file():
                continue

            try:
                resolved_path = path.resolve(strict=False)
            except OSError:
                resolved_path = path.absolute()

            if _has_hidden_component(resolved_path):
                continue

            if _path_in(resolved_path, excluded_paths):
                continue

            if resolved_path.suffix.lower() not in exts:
                continue

            yield resolved_path


__all__ = ["iter_images"]
