"""Tests for filesystem utility helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ctypes
import pytest

from utils import fs


def test_to_system_path_long_paths_receive_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs, "WINDOWS", True)
    segment = "a" * 260
    candidate = tmp_path / segment
    expected = f"{fs.LONG_PATH_PREFIX}{candidate.resolve()}"

    result = fs.to_system_path(candidate)

    assert result == expected


def test_to_system_path_short_paths_preserve_original(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs, "WINDOWS", True)
    candidate = tmp_path / "short.txt"

    result = fs.to_system_path(candidate)

    assert result == str(candidate.resolve())


def test_to_system_path_retains_existing_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs, "WINDOWS", True)
    monkeypatch.setattr(fs.Path, "resolve", lambda self, strict=False: self)
    original = f"{fs.LONG_PATH_PREFIX}C:\\example\\file.txt"
    candidate = Path(original)

    result = fs.to_system_path(candidate)

    assert result == original


def test_from_system_path_strips_prefix_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs, "WINDOWS", True)
    target = tmp_path / "sample.txt"
    prefixed = f"{fs.LONG_PATH_PREFIX}{target}"

    result = fs.from_system_path(prefixed)

    assert result == target


def test_from_system_path_returns_path_unchanged_when_not_prefixed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs, "WINDOWS", True)
    target = tmp_path / "regular.txt"

    result = fs.from_system_path(str(target))

    assert result == target


def test_is_hidden_detects_dotfiles(tmp_path: Path) -> None:
    hidden = tmp_path / ".secret"
    hidden.touch()

    assert fs.is_hidden(hidden) is True


def test_is_hidden_checks_windows_attributes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs, "WINDOWS", True)
    attrs = 0x2
    dummy_kernel32 = SimpleNamespace(GetFileAttributesW=lambda _: attrs)
    dummy_windll = SimpleNamespace(kernel32=dummy_kernel32)
    monkeypatch.setattr(ctypes, "windll", dummy_windll, raising=False)
    candidate = tmp_path / "hidden.txt"

    assert fs.is_hidden(candidate) is True


def test_is_hidden_returns_false_on_ctypes_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs, "WINDOWS", True)

    class FaultyKernel32:
        def GetFileAttributesW(self, _: str) -> int:
            raise RuntimeError("boom")

    dummy_windll = SimpleNamespace(kernel32=FaultyKernel32())
    monkeypatch.setattr(ctypes, "windll", dummy_windll, raising=False)
    candidate = tmp_path / "not_hidden.txt"

    assert fs.is_hidden(candidate) is False


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("a", "inside.txt"), True),
        (("b", "sub", "file.txt"), True),
        (("c", "outside.txt"), False),
        (("..", "external.txt"), False),
    ],
)
def test_path_in_roots(tmp_path: Path, parts: tuple[str, ...], expected: bool) -> None:
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()

    candidate = tmp_path.joinpath(*parts)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.touch()

    assert fs.path_in_roots(candidate, roots) is expected
