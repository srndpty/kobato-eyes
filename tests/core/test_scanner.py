"""Tests for filesystem scanning utilities."""

from __future__ import annotations

from pathlib import Path

from core.scanner import iter_images


def test_iter_images_filters_hidden_and_excluded(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    (root / "visible.jpg").write_bytes(b"test")
    (root / "document.txt").write_bytes(b"ignored")

    hidden_dir = root / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.png").write_bytes(b"secret")

    hidden_file = root / ".hidden_file.png"
    hidden_file.write_bytes(b"secret")

    excluded_dir = root / "skip"
    excluded_dir.mkdir()
    (excluded_dir / "ignored.jpg").write_bytes(b"skip")

    subdir = root / "sub"
    subdir.mkdir()
    (subdir / "photo.PNG").write_bytes(b"data")

    images = list(
        iter_images(
            [root],
            excluded=[excluded_dir],
            extensions=["jpg", "png"],
        )
    )

    assert sorted(path.name for path in images) == ["photo.PNG", "visible.jpg"]


def test_iter_images_handles_deep_paths(tmp_path: Path) -> None:
    root = tmp_path / ("a" * 10)
    current = root
    for _ in range(10):
        current.mkdir(exist_ok=True)
        current = current / ("b" * 10)
    current.parent.mkdir(parents=True, exist_ok=True)
    image_path = current.parent / "deep.png"
    image_path.write_bytes(b"data")

    results = list(iter_images([root], extensions=["png"]))
    assert results and results[0].name == "deep.png"


def test_iter_images_handles_parentheses_in_path(tmp_path: Path) -> None:
    root = tmp_path / "root (test)"
    nested = root / "album (1)"
    nested.mkdir(parents=True)
    image_path = nested / "photo.JPG"
    image_path.write_bytes(b"data")

    results = list(iter_images([root], extensions=["jpg"]))

    assert image_path in results
