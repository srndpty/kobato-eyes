from __future__ import annotations

import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

pixai_onnx = pytest.importorskip(
    "tagger.pixai_onnx", reason="Pixai tagger requires optional dependencies", exc_type=ModuleNotFoundError
)
from tagger.base import TagCategory  # noqa: E402 - imported after optional dependency resolution


np = pixai_onnx.np


@pytest.fixture(autouse=True)
def _mock_pixai_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide a lightweight fake onnxruntime implementation for Pixai tests."""

    class _DummySession:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
            self._inputs = [SimpleNamespace(name="input_0", shape=[None, 512, 512, 3])]
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
            return str(tmp_path / "kobato-eyes" / "logs" / "pixai_profile.json")

        def run(self, output_names, feeds):  # noqa: D401 - signature compatibility
            return [np.array([[0.9, 0.85, 0.2]], dtype=np.float32)]

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
    monkeypatch.setattr(pixai_onnx.wd14_onnx, "ort", _DummyOrt)
    monkeypatch.setattr(pixai_onnx.wd14_onnx, "_IMPORT_ERROR", None)
    monkeypatch.setattr(pixai_onnx.wd14_onnx, "_ACTIVE_TAGGERS", weakref.WeakSet())


def test_pixai_merges_copyright_tags(tmp_path: Path) -> None:
    model_path = tmp_path / "pixai.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,tag_id,name,category,count,ips",
                "0,1,general_tag,0,1000,[]",
                '1,2,character_a,1,500,"[""series_a"", ""series_b""]"',
                "2,3,series_a,3,400,[]",
            ]
        ),
        encoding="utf-8",
    )

    tagger = pixai_onnx.PixaiOnnxTagger(model_path, tags_csv=csv_path)

    assert tagger.input_size_px == 512

    batch = np.zeros((1, tagger.input_size_px, tagger.input_size_px, 3), dtype=np.float32)
    results = tagger.infer_batch_prepared(
        batch,
        thresholds={
            TagCategory.GENERAL: 0.0,
            TagCategory.CHARACTER: 0.0,
            TagCategory.COPYRIGHT: 0.0,
        },
    )

    assert len(results) == 1
    result = results[0]
    names = [prediction.name for prediction in result.tags]
    assert names == ["general_tag", "character_a", "series_a", "series_b"]
    scores = {prediction.name: prediction.score for prediction in result.tags if prediction.category == TagCategory.COPYRIGHT}
    assert scores["series_a"] == pytest.approx(0.85)
    assert scores["series_b"] == pytest.approx(0.85)
