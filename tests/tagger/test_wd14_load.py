"""Tests for WD14 ONNX tagger configuration handling."""

from __future__ import annotations

import logging
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from tagger import wd14_onnx
from tagger.base import TagCategory


@pytest.fixture(autouse=True)
def _mock_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide a lightweight fake onnxruntime implementation for tests."""

    class _DummySession:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
            self._inputs = [SimpleNamespace(name="input_0")]
            self._outputs = [SimpleNamespace(name="output_0")]
            self._providers = list(kwargs.get("providers", []))
            self._provider_options = {provider: {} for provider in self._providers}
            self._sess_options = kwargs.get("sess_options")

        def get_inputs(self) -> list[SimpleNamespace]:
            return self._inputs

        def get_outputs(self) -> list[SimpleNamespace]:
            return self._outputs

        def get_providers(self) -> list[str]:
            return list(self._providers)

        def get_provider_options(self) -> dict[str, dict[str, object]]:
            return dict(self._provider_options)

        def end_profiling(self) -> str:
            return str(tmp_path / "kobato-eyes" / "logs" / "wd14_profile.json")

    class _DummyOrt:
        class SessionOptions:  # noqa: D401 - simple placeholder
            def __init__(self) -> None:
                self.graph_optimization_level = None
                self.enable_profiling = False
                self.log_severity_level = 1
                self.profile_file_prefix = ""

        class GraphOptimizationLevel:  # noqa: D401 - simple placeholder
            ORT_ENABLE_ALL = 99

        InferenceSession = _DummySession

        @staticmethod
        def get_available_providers() -> list[str]:  # noqa: D401 - simple placeholder
            return ["CPUExecutionProvider"]

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(wd14_onnx, "ort", _DummyOrt)
    monkeypatch.setattr(wd14_onnx, "_IMPORT_ERROR", None)
    monkeypatch.setattr(wd14_onnx, "_ACTIVE_TAGGERS", weakref.WeakSet())


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

    session_options = tagger._session._sess_options  # type: ignore[attr-defined]
    # assert session_options.enable_profiling is True # $env:KE_ORT_PROFILEの値次第なのでここはチェックしない
    assert session_options.log_severity_level == 2
    assert session_options.graph_optimization_level == 99
    assert Path(session_options.profile_file_prefix).name.startswith("wd14")


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
            self._providers = list(providers)
            self._provider_options = {provider: {} for provider in self._providers}
            self._sess_options = kwargs.get("sess_options")

        def get_inputs(self) -> list[SimpleNamespace]:
            return self._inputs

        def get_outputs(self) -> list[SimpleNamespace]:
            return self._outputs

        def get_providers(self) -> list[str]:
            return list(self._providers)

        def get_provider_options(self) -> dict[str, dict[str, object]]:
            return dict(self._provider_options)

        def end_profiling(self) -> str:
            return "dummy"

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
            self._providers = list(providers)
            self._provider_options = {provider: {} for provider in self._providers}
            self._sess_options = kwargs.get("sess_options")

        def get_inputs(self) -> list[SimpleNamespace]:
            return self._inputs

        def get_outputs(self) -> list[SimpleNamespace]:
            return self._outputs

        def get_providers(self) -> list[str]:
            return list(self._providers)

        def get_provider_options(self) -> dict[str, dict[str, object]]:
            return dict(self._provider_options)

        def end_profiling(self) -> str:
            return "dummy"

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


def test_end_profile_writes_path(tmp_path: Path, _mock_labels: list[Path]) -> None:
    """Calling end_profile should return the profiling path."""

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("tag,0\n", encoding="utf-8")

    tagger = wd14_onnx.WD14Tagger(model_path)
    profile_path = tagger.end_profile()
    assert profile_path is not None
    assert profile_path.endswith("wd14_profile.json")


def test_end_all_profiles_invokes_each(
    tmp_path: Path, _mock_labels: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper should close profiling for all tracked taggers."""

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("tag,0\n", encoding="utf-8")

    tagger = wd14_onnx.WD14Tagger(model_path)
    other = wd14_onnx.WD14Tagger(model_path)
    assert len(list(wd14_onnx._ACTIVE_TAGGERS)) == 2  # type: ignore[attr-defined]
    calls: list[int] = []

    def _spy(self: wd14_onnx.WD14Tagger) -> str | None:  # type: ignore[override]
        calls.append(id(self))
        return "dummy"

    monkeypatch.setattr(wd14_onnx.WD14Tagger, "end_profile", _spy, raising=False)
    wd14_onnx.end_all_profiles()
    assert calls.count(id(tagger)) == 1
    assert calls.count(id(other)) == 1
