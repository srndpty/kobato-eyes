from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

np = pytest.importorskip("numpy")

from tagger.base import TagCategory


@pytest.fixture(autouse=True)
def _mock_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import tagger.pixai_onnx as pixai_onnx
    from tagger import wd14_onnx

    run_calls: list[tuple[list[str], tuple[int, ...]]] = []

    class _DummySession:
        def __init__(self, *args, **kwargs) -> None:
            self._inputs = [SimpleNamespace(name="input", shape=[None, 3, 512, 512])]
            self._outputs = [
                SimpleNamespace(name="embedding", shape=[None, 16]),
                SimpleNamespace(name="logits", shape=[None, 4]),
                SimpleNamespace(name="probabilities", shape=[None, 4]),
            ]
            self._providers = list(kwargs.get("providers", []))
            self._provider_options = {provider: {} for provider in self._providers}
            self._sess_options = kwargs.get("sess_options")

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def get_providers(self):
            return list(self._providers)

        def get_provider_options(self):
            return dict(self._provider_options)

        def end_profiling(self) -> str:
            return str(tmp_path / "kobato-eyes" / "logs" / "pixai_profile.json")

        def run(self, output_names, feeds):
            batch = next(iter(feeds.values()))
            run_calls.append((list(output_names), tuple(batch.shape)))
            logits = np.array([[0.2, 4.5, -4.0, 2.5]], dtype=np.float32)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            tensor_map = {
                "embedding": np.zeros((batch.shape[0], 16), dtype=np.float32),
                "logits": logits,
                "probabilities": probabilities,
            }
            return [tensor_map[name] for name in output_names]

    class _DummyOrt:
        class SessionOptions:
            def __init__(self) -> None:
                self.graph_optimization_level = None
                self.enable_profiling = False
                self.log_severity_level = 1
                self.profile_file_prefix = ""

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = 99

        InferenceSession = _DummySession

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(wd14_onnx, "ort", _DummyOrt)
    monkeypatch.setattr(pixai_onnx, "ort", _DummyOrt)
    monkeypatch.setattr(wd14_onnx, "_IMPORT_ERROR", None)
    monkeypatch.setattr(pixai_onnx, "_IMPORT_ERROR", None)
    monkeypatch.setattr(wd14_onnx, "_ACTIVE_TAGGERS", weakref.WeakSet())
    return run_calls


def test_pixai_adds_copyright_from_ips(tmp_path: Path, _mock_ort):
    from tagger.pixai_onnx import PixaiOnnxTagger

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"pixai")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,tag_id,name,category,count,ips",
                "0,1,general_tag,0,100,[]",
                '1,2,heroine,1,50,"[\"franchise\"]"',
                "2,3,,0,0,[]",
                "3,4,another,3,10,[]",
            ]
        ),
        encoding="utf-8",
    )

    tagger = PixaiOnnxTagger(model_path, tags_csv=csv_path)
    batch = np.zeros((1, 512, 512, 3), dtype=np.float32)
    result = tagger.infer_batch_prepared(batch)[0]

    names = {tag.name: tag for tag in result.tags}
    assert "heroine" in names
    assert "franchise" in names
    assert names["franchise"].category == TagCategory.COPYRIGHT
    assert names["franchise"].score == pytest.approx(names["heroine"].score)
    assert "" not in names

    output_names, input_shape = _mock_ort[-1]
    assert output_names == ["logits"]
    assert input_shape == (1, 3, 512, 512)
