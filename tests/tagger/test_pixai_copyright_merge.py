from __future__ import annotations

import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tagger import pixai_onnx, wd14_onnx
from tagger.base import TagCategory


@pytest.fixture(autouse=True)
def _mock_pixai_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide a lightweight ONNX Runtime substitute for PixAI tests."""

    class _DummySession:
        def __init__(self, *args, **kwargs) -> None:
            self._inputs = [SimpleNamespace(name="input_0", shape=[None, 3, 448, 448])]
            label_count = 4
            self._outputs = [
                SimpleNamespace(name="embedding", shape=[None, 1280]),
                SimpleNamespace(name="probabilities", shape=[None, label_count]),
                SimpleNamespace(name="logits", shape=[None, label_count]),
            ]
            self._providers = list(kwargs.get("providers", []))
            self._provider_options = {provider: {} for provider in self._providers}

        def get_inputs(self) -> list[SimpleNamespace]:  # noqa: D401 - simple stub
            return self._inputs

        def get_outputs(self) -> list[SimpleNamespace]:  # noqa: D401 - simple stub
            return self._outputs

        def get_providers(self) -> list[str]:  # noqa: D401 - simple stub
            return list(self._providers)

        def get_provider_options(self) -> dict[str, dict[str, object]]:  # noqa: D401 - simple stub
            return dict(self._provider_options)

        def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            batch = next(iter(feeds.values()))
            batch_size = batch.shape[0]
            logits = np.array([[1.5, 3.0, -5.0, -2.0]], dtype=np.float32)
            probs = 1.0 / (1.0 + np.exp(-logits))
            embedding = np.zeros((batch_size, 1280), dtype=np.float32)
            tiled_logits = np.repeat(logits, batch_size, axis=0)
            tiled_probs = np.repeat(probs, batch_size, axis=0)
            return [embedding, tiled_probs, tiled_logits]

        def end_profiling(self) -> str:  # noqa: D401 - simple stub
            return str(tmp_path / "kobato-eyes" / "logs" / "pixai_profile.json")

    class _DummyOrt:
        class SessionOptions:  # noqa: D401 - simple placeholder
            def __init__(self) -> None:
                self.graph_optimization_level = None
                self.enable_profiling = True
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


def test_pixai_tagger_merges_ips(tmp_path: Path) -> None:
    """PixAI tagger should add IPS copyright tags for character predictions."""

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dummy")
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        (
            "id,tag_id,name,category,count,ips\n"
            "0,1,tag-general,0,10,[]\n"
            '1,2,heroine,1,5,"[""series-a"", ""series-b""]"\n'
            "2,3,,1,0,[]\n"
            "3,4,series-a,3,2,[]\n"
        ),
        encoding="utf-8",
    )

    tagger = pixai_onnx.PixaiOnnxTagger(model_path, tags_csv=csv_path)
    batch = np.zeros((1, 448, 448, 3), dtype=np.float32)
    results = tagger.infer_batch_prepared(batch)

    assert len(results) == 1
    names = {tag.name: tag for tag in results[0].tags}
    assert "heroine" in names
    assert names["heroine"].category is TagCategory.CHARACTER
    assert names["series-a"].category is TagCategory.COPYRIGHT
    assert names["series-b"].category is TagCategory.COPYRIGHT
    assert "" not in names
