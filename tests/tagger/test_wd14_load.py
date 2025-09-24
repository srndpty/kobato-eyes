"""Tests for WD14 ONNX tagger configuration handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tagger import wd14_onnx
from tagger.base import TagCategory


@pytest.fixture(autouse=True)
def _mock_ort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a lightweight fake onnxruntime implementation for tests."""

    class _DummySession:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
            self._inputs = [SimpleNamespace(name="input_0")]
            self._outputs = [SimpleNamespace(name="output_0")]

        def get_inputs(self) -> list[SimpleNamespace]:
            return self._inputs

        def get_outputs(self) -> list[SimpleNamespace]:
            return self._outputs

    class _DummyOrt:
        class SessionOptions:  # noqa: D401 - simple placeholder
            def __init__(self) -> None:
                pass

        InferenceSession = _DummySession

    monkeypatch.setattr(wd14_onnx, "ort", _DummyOrt)
    monkeypatch.setattr(wd14_onnx, "_IMPORT_ERROR", None)


@pytest.fixture
def _mock_labels(monkeypatch: pytest.MonkeyPatch):
    """Track the label CSV path WD14Tagger attempts to load."""
    captured: list[Path] = []

    def _fake_loader(path: str | Path) -> list[wd14_onnx._Label]:  # type: ignore[name-defined]
        captured.append(Path(path))
        return [wd14_onnx._Label(name="tag", category=TagCategory.GENERAL)]

    monkeypatch.setattr(wd14_onnx.WD14Tagger, "_load_labels", staticmethod(_fake_loader))
    return captured


def test_wd14_tagger_auto_discovers_csv(tmp_path: Path, _mock_labels: list[Path]) -> None:
    """WD14Tagger should locate selected_tags.csv next to the model."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("tag,0\n", encoding="utf-8")

    tagger = wd14_onnx.WD14Tagger(model_path)

    assert _mock_labels[0] == csv_path
    assert tagger._labels_path == csv_path  # type: ignore[attr-defined]


def test_wd14_tagger_respects_explicit_csv(tmp_path: Path, _mock_labels: list[Path]) -> None:
    """WD14Tagger should honour an explicit tags_csv path."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "custom.csv"
    csv_path.write_text("tag,0\n", encoding="utf-8")

    tagger = wd14_onnx.WD14Tagger(model_path, tags_csv=csv_path)

    assert _mock_labels[0] == csv_path
    assert tagger._labels_path == csv_path  # type: ignore[attr-defined]
