"""Property-based tests for :func:`core.scanner.iter_images` with Windows-like paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from core.scanner import DEFAULT_EXTENSIONS, iter_images

WINDOWS_INVALID_CHARS = "<>:\"/\\|?*"
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"]


@dataclass
class FileEntry:
    """Parameters describing a file to create for property testing."""

    segments: list[tuple[str, bool]]
    filename: str
    extra_suffix: str | None
    file_hidden: bool
    extension: str
    exclude_dir: bool
    exclude_file: bool
    create_extra_file: bool


def _segment_characters() -> st.SearchStrategy[str]:
    """Return a strategy generating characters valid for Windows filenames."""

    return st.characters(
        whitelist_categories=(
            "Lu",
            "Ll",
            "Lt",
            "Lm",
            "Lo",
            "Mn",
            "Mc",
            "Nd",
            "Nl",
            "No",
            "Pc",
            "Pd",
            "Ps",
            "Pe",
            "Pi",
            "Pf",
            "Po",
            "Sm",
            "Sc",
            "Sk",
            "So",
            "Zs",
        ),
        blacklist_characters=WINDOWS_INVALID_CHARS + "\x00",
    )


def _valid_segment(text: str) -> bool:
    """Check whether *text* is safe to use as a Windows path segment."""

    if not text:
        return False
    if text in {".", ".."}:
        return False
    if text[-1] in {" ", "."}:
        return False
    if all(char.isspace() for char in text):
        return False
    if any(char in WINDOWS_INVALID_CHARS for char in text):
        return False
    return True


segment_strategy = st.text(_segment_characters(), min_size=1, max_size=12).filter(_valid_segment)


file_entry_strategy = st.builds(
    FileEntry,
    segments=st.lists(st.tuples(segment_strategy, st.booleans()), min_size=0, max_size=3),
    filename=segment_strategy,
    extra_suffix=st.one_of(st.none(), segment_strategy),
    file_hidden=st.booleans(),
    extension=st.tuples(st.sampled_from(ALLOWED_EXTENSIONS), st.booleans()).map(
        lambda data: data[0].upper() if data[1] else data[0].lower()
    ),
    exclude_dir=st.booleans(),
    exclude_file=st.booleans(),
    create_extra_file=st.booleans(),
)


extension_token_strategy = st.builds(
    lambda ext, dotted, upper: (f".{ext.upper()}" if upper else f".{ext.lower()}") if dotted else (ext.upper() if upper else ext.lower()),
    ext=st.sampled_from(ALLOWED_EXTENSIONS),
    dotted=st.booleans(),
    upper=st.booleans(),
)


extensions_strategy = st.one_of(
    st.none(),
    st.lists(extension_token_strategy, min_size=1, max_size=len(ALLOWED_EXTENSIONS)),
)


def _resolve(path: Path) -> Path:
    """Resolve *path* as :func:`core.scanner.iter_images` does."""

    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _has_hidden_component(path: Path) -> bool:
    """Determine whether *path* contains hidden components (dot-prefixed)."""

    for part in path.parts:
        if part.startswith(".") and part not in {".", ".."}:
            return True
    return False


def _path_in(path: Path, bases: Iterable[Path]) -> bool:
    """Return ``True`` if *path* is located under any of *bases*."""

    for base in bases:
        try:
            path.relative_to(base)
        except ValueError:
            continue
        return True
    return False


@given(entries=st.lists(file_entry_strategy, min_size=1, max_size=5), extensions=extensions_strategy)
@settings(max_examples=75, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@example(
    entries=[
        FileEntry(
            segments=[("ハート♡と日本語と空白", False)],
            filename="love",
            extra_suffix=None,
            file_hidden=False,
            extension="JPG",
            exclude_dir=False,
            exclude_file=False,
            create_extra_file=False,
        ),
        FileEntry(
            segments=[("emoji", False), ("🧡mix", False)],
            filename="smile",
            extra_suffix="v1",
            file_hidden=False,
            extension="png",
            exclude_dir=True,
            exclude_file=False,
            create_extra_file=True,
        ),
    ],
    extensions=["jpg", ".PNG"],
)
@example(
    entries=[
        FileEntry(
            segments=[("مرحبا", False)],
            filename="greeting",
            extra_suffix=None,
            file_hidden=False,
            extension="tiff",
            exclude_dir=False,
            exclude_file=False,
            create_extra_file=False,
        ),
        FileEntry(
            segments=[("Привет", False)],
            filename="мир",
            extra_suffix=None,
            file_hidden=True,
            extension="GIF",
            exclude_dir=False,
            exclude_file=True,
            create_extra_file=False,
        ),
    ],
    extensions=None,
)
def test_iter_images_property_based(entries: list[FileEntry], extensions: list[str] | None, tmp_path: Path) -> None:
    """Property test ensuring specification-compliant behaviour for ``iter_images``."""

    created_files: list[Path] = []
    excluded_arguments: list[str | Path] = []

    for entry in entries:
        directory = tmp_path
        for segment, make_hidden in entry.segments:
            name = segment
            if make_hidden and not name.startswith("."):
                name = f".{name}"
            directory = directory / name
        directory.mkdir(parents=True, exist_ok=True)

        if entry.create_extra_file:
            (directory / "notes.txt").write_text("note", encoding="utf-8")

        name_body = entry.filename
        if entry.extra_suffix:
            name_body = f"{name_body}.{entry.extra_suffix}"
        if entry.file_hidden and not name_body.startswith("."):
            name_body = f".{name_body}"

        file_path = directory / f"{name_body}.{entry.extension}"
        assume(len(str(file_path)) < 240)
        file_path.write_bytes(b"data")
        created_files.append(file_path)

        if entry.exclude_dir:
            excluded_arguments.append(directory)
        if entry.exclude_file:
            excluded_arguments.append(file_path)

    # Always include a non-existent exclusion to confirm robustness.
    excluded_arguments.append(tmp_path / "does-not-exist")

    # Mix string and Path inputs for excluded paths.
    excluded_mixed: list[str | Path] = []
    for index, item in enumerate(excluded_arguments):
        excluded_mixed.append(str(item) if index % 2 == 0 else item)

    roots = [tmp_path, tmp_path / "missing-root"]

    if extensions is None:
        extensions_arg = None
        expected_exts = DEFAULT_EXTENSIONS
    else:
        extensions_arg = extensions
        expected_exts = {
            (ext.lower() if ext.startswith(".") else f".{ext.lower()}")
            for ext in extensions
        }

    excluded_resolved = [_resolve(Path(item)) for item in excluded_mixed]

    expected: set[Path] = set()
    for file_path in created_files:
        resolved = _resolve(file_path)
        if _has_hidden_component(resolved):
            continue
        if _path_in(resolved, excluded_resolved):
            continue
        if resolved.suffix.lower() not in expected_exts:
            continue
        expected.add(resolved)

    results = set(iter_images(roots, excluded=excluded_mixed, extensions=extensions_arg))

    missing = expected - results
    unexpected = results - expected
    assert not missing and not unexpected, (
        "Property mismatch. "
        f"Missing: {sorted(missing)}; Unexpected: {sorted(unexpected)}"
    )
