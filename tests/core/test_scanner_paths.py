"""Example-based tests for :func:`core.scanner.iter_images` path handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.scanner import iter_images


def _resolve(path: Path) -> Path:
    """Resolve *path* similarly to :func:`core.scanner.iter_images`."""

    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _is_under(path: Path, base: Path) -> bool:
    """Return ``True`` if *path* is located under *base*."""

    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def test_iter_images_handles_special_path_names(tmp_path: Path) -> None:
    """Iterating finds files in directories with diverse Unicode and symbols."""

    special_dirs = [
        "space name",
        "paren (left) and (right)",
        "[brackets] {braces}",
        "symbols & + , ; = ! # $ % ^ ~ @`",
        "multi.dots.name.image.JPG",
        "ハート♡と日本語と空白",
        "한글 테스트",
        "Привет-мир",
        "مرحبا",
        "e\u0301とé（合成／単一）",
        "emoji_🧡_nonbmp",
        "skip (♥) [x]",
    ]
    excluded_names = {"symbols & + , ; = ! # $ % ^ ~ @`", "skip (♥) [x]"}

    created_files: list[Path] = []
    for index, folder_name in enumerate(special_dirs):
        folder = tmp_path / folder_name
        folder.mkdir()

        primary = folder / f"image_{index}.{'JPG' if index % 2 == 0 else 'png'}"
        primary.write_bytes(b"test")
        created_files.append(primary)

        if index % 3 == 0:
            secondary = folder / f"extra_{index}.PnG"
            secondary.write_bytes(b"extra")
            created_files.append(secondary)

        (folder / f"note_{index}.txt").write_text("ignore me", encoding="utf-8")

    excluded_paths = [tmp_path / name for name in excluded_names]
    excluded_argument: list[str | Path] = []
    for offset, excluded in enumerate(excluded_paths):
        excluded_argument.append(str(excluded) if offset % 2 == 0 else excluded)

    missing_root = tmp_path / "does-not-exist"
    results = set(
        iter_images([missing_root, tmp_path], excluded=excluded_argument)
    )

    excluded_resolved = [_resolve(path) for path in excluded_paths]
    expected = set()
    for file_path in created_files:
        resolved_file = _resolve(file_path)
        if any(
            _is_under(resolved_file, excluded_root) or resolved_file == excluded_root
            for excluded_root in excluded_resolved
        ):
            continue
        expected.add(resolved_file)

    assert results == expected, (
        "Iterated files mismatch. "
        f"Missing: {sorted(expected - results)}; "
        f"Unexpected: {sorted(results - expected)}"
    )


def test_iter_images_skips_hidden_components(tmp_path: Path) -> None:
    """Hidden directories and files prefixed with a dot are ignored."""

    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    nested_hidden = hidden_dir / "sub"
    nested_hidden.mkdir()
    secret = nested_hidden / "secret.png"
    secret.write_bytes(b"secret")

    hidden_file = tmp_path / ".hidden_file.jpg"
    hidden_file.write_bytes(b"hidden")

    visible = tmp_path / "visible.JPG"
    visible.write_bytes(b"visible")

    discovered = set(iter_images([tmp_path]))
    resolved_visible = _resolve(visible)

    assert resolved_visible in discovered, "Visible image should be listed."
    assert _resolve(secret) not in discovered, "Hidden directory contents must be excluded."
    assert _resolve(hidden_file) not in discovered, "Hidden files must be excluded."


def test_iter_images_handles_long_nested_paths(tmp_path: Path) -> None:
    """Deeply nested paths beyond the legacy 260-character limit are supported when possible."""

    nested = tmp_path
    for index in range(12):
        nested = nested / f"level_{index:02d}_{'a' * 20}"
        try:
            nested.mkdir()
        except OSError as exc:  # pragma: no cover - platform dependent
            pytest.skip(f"Filesystem cannot create deep directories: {exc}")

    deep_file = nested / "deep_image.PNG"
    try:
        deep_file.write_bytes(b"deep")
    except OSError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"Filesystem cannot create files at deep paths: {exc}")

    if len(str(deep_file)) <= 260:
        pytest.skip("Generated path does not exceed 260 characters; skipping long-path check.")

    discovered = set(iter_images([tmp_path]))
    assert _resolve(deep_file) in discovered, f"Deep path {deep_file} was not detected."


def test_iter_images_honours_mixed_case_extension_filters(tmp_path: Path) -> None:
    """Extension filters accept mixed casing and optional leading dots."""

    files = {
        "photo.PNG": True,
        "scan.JpG": True,
        "icon.webp": True,
        "notes.txt": False,
        "archive.tgz": False,
    }

    for name, is_image in files.items():
        path = tmp_path / name
        if is_image:
            path.write_bytes(b"image")
        else:
            path.write_text("text", encoding="utf-8")

    filtered = set(iter_images([tmp_path], extensions=["jpg", "PNG", ".webp"]))
    expected = {
        _resolve(tmp_path / name)
        for name, is_image in files.items()
        if is_image
    }

    assert filtered == expected, "Mixed-case extension filtering failed."


def test_iter_images_excludes_paths_with_special_characters(tmp_path: Path) -> None:
    """Paths supplied via *excluded* are skipped even with complex characters."""

    include_dir = tmp_path / "keep-me"
    include_dir.mkdir()
    keep_file = include_dir / "kept_image.jpg"
    keep_file.write_bytes(b"keep")

    excluded_dir = tmp_path / "exclude (♥) [100%]"
    excluded_dir.mkdir()
    drop_file = excluded_dir / "drop.png"
    drop_file.write_bytes(b"drop")

    results = set(
        iter_images(
            [tmp_path],
            excluded=[str(excluded_dir)],
        )
    )

    assert _resolve(keep_file) in results, "Non-excluded images should remain."
    assert _resolve(drop_file) not in results, "Excluded directory contents must be skipped."
