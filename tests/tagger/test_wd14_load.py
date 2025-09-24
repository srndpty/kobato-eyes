"""Tests for WD14 ONNX tagger configuration handling."""

from __future__ import annotations

import logging
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

        @staticmethod
        def get_available_providers() -> list[str]:  # noqa: D401 - simple placeholder
            return ["CPUExecutionProvider"]

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


def test_wd14_tagger_falls_back_to_cpu_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    _mock_labels: list[Path],
) -> None:
    """WD14Tagger should gracefully fall back to CPU provider when CUDA fails."""

    attempts: list[list[str]] = []

    class _FailCUDAThenCPU:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
            providers = kwargs.get("providers", [])
            attempts.append(list(providers))
            if providers == ["CUDAExecutionProvider"]:
                raise RuntimeError("CUDAExecutionProvider not available")
            self._inputs = [SimpleNamespace(name="input_0")]
            self._outputs = [SimpleNamespace(name="output_0")]

        def get_inputs(self) -> list[SimpleNamespace]:
            return self._inputs

        def get_outputs(self) -> list[SimpleNamespace]:
            return self._outputs

    monkeypatch.setattr(wd14_onnx.ort, "InferenceSession", _FailCUDAThenCPU)

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("tag,0\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        tagger = wd14_onnx.WD14Tagger(model_path)

    assert attempts[0] == ["CUDAExecutionProvider"]
    assert attempts[1] == ["CPUExecutionProvider"]
    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert any("falling back to CPUExecutionProvider" in message for message in warning_messages)
    assert tagger._session is not None  # type: ignore[attr-defined]


def test_wd14_tagger_warns_when_cuda_requested_but_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    _mock_labels: list[Path],
) -> None:
    """Explicit CUDA requests should fall back to CPU when unavailable."""

    attempts: list[list[str]] = []

    class _RecordProviders:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
            providers = kwargs.get("providers", [])
            attempts.append(list(providers))
            self._inputs = [SimpleNamespace(name="input_0")]
            self._outputs = [SimpleNamespace(name="output_0")]

        def get_inputs(self) -> list[SimpleNamespace]:
            return self._inputs

        def get_outputs(self) -> list[SimpleNamespace]:
            return self._outputs

    def _fake_available() -> list[str]:
        return ["CPUExecutionProvider"]

    monkeypatch.setattr(wd14_onnx.ort, "InferenceSession", _RecordProviders)
    monkeypatch.setattr(wd14_onnx, "get_available_providers", _fake_available)

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("tag,0\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        tagger = wd14_onnx.WD14Tagger(model_path, providers=["CUDAExecutionProvider"])

    assert attempts == [["CPUExecutionProvider"]]
    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert any("requested but not available" in message for message in warning_messages)
    assert tagger._session is not None  # type: ignore[attr-defined]
