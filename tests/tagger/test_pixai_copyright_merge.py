from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
np = pytest.importorskip("numpy")

from tagger import wd14_onnx
from tagger.base import TagCategory
from tagger.pixai_onnx import PixaiOnnxTagger


@pytest.fixture(autouse=True)
def _mock_pixai_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide a lightweight fake onnxruntime implementation for PixAI tests."""

    class _DummySession:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
            self._inputs = [SimpleNamespace(name="input_0", shape=[1, 3, 448, 448])]
            self._outputs = [
                SimpleNamespace(name="embedding", shape=[1, 1280]),
                SimpleNamespace(name="logits", shape=[1, 4]),
                SimpleNamespace(name="prediction", shape=[1, 4]),
            ]
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

        def run(self, output_names, feeds):  # noqa: D401 - signature compatibility
            batch = next(iter(feeds.values()))
            batch_size = batch.shape[0]
            return [np.zeros((batch_size, 4), dtype=np.float32) for _ in output_names]

        def end_profiling(self) -> str:
            return str(tmp_path / "kobato-eyes" / "logs" / "pixai_profile.json")

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


def test_pixai_tagger_merges_ips_as_copyright(tmp_path: Path) -> None:
    """PixAI should emit copyright tags from ips metadata."""

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        """id,tag_id,name,category,count,ips
0,1,tag_general,0,10,[]
1,2,tag_character,1,5,"[\"series_a\"]"
2,3,,1,0,[]
3,4,tag_copyright,3,2,[]
""",
        encoding="utf-8",
    )

    tagger = PixaiOnnxTagger(model_path, tags_csv=csv_path)

    logits = np.array([[0.9, 0.95, 0.8, 0.85]], dtype=np.float32)
    results = tagger._postprocess_logits_topk(logits, thresholds=None, max_tags=None)
    assert len(results) == 1
    predictions = results[0].tags

    names = {pred.name: pred for pred in predictions}
    assert "" not in names
    assert "tag_character" in names
    assert names["tag_character"].category == TagCategory.CHARACTER
    assert "tag_copyright" in names
    assert names["tag_copyright"].category == TagCategory.COPYRIGHT
    assert "series_a" in names
    assert names["series_a"].category == TagCategory.COPYRIGHT
    assert np.isclose(names["series_a"].score, names["tag_character"].score)

