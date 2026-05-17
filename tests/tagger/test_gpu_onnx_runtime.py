"""GPU smoke tests for ONNX Runtime tagger execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tagger.onnx_backend import CUDA_PROVIDER
from tagger.wd14_onnx import WD14Tagger

pytestmark = pytest.mark.gpu


def _cuda_ort():
    """Return onnxruntime when CUDAExecutionProvider is available."""

    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    providers = list(ort.get_available_providers())
    if CUDA_PROVIDER not in providers:
        pytest.skip(f"{CUDA_PROVIDER} is not available; providers={providers}")
    return ort


def test_onnxruntime_cuda_provider_is_available() -> None:
    """Ensure GPU check collects a concrete CUDA provider smoke test."""

    ort = _cuda_ort()

    assert CUDA_PROVIDER in ort.get_available_providers()


def test_wd14_cuda_dummy_model_smoke(tmp_path: Path) -> None:
    """Run WD14Tagger against a tiny ONNX model through CUDA provider."""

    _cuda_ort()
    onnx = pytest.importorskip("onnx", reason="onnx package is required to build the dummy model")
    from onnx import TensorProto, helper

    model_path = tmp_path / "dummy-wd14.onnx"
    labels_path = tmp_path / "selected_tags.csv"
    labels_path.write_text("tag_a,0\ntag_b,0\n", encoding="utf-8")

    input_info = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 4, 4, 3])
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    constant = helper.make_tensor(
        "logits_const",
        TensorProto.FLOAT,
        [1, 2],
        [0.25, 0.95],
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", inputs=[], outputs=["logits"], value=constant)],
        "dummy_wd14",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)

    tagger = WD14Tagger(
        model_path,
        tags_csv=labels_path,
        providers=[CUDA_PROVIDER],
        input_size=4,
        default_thresholds={},
    )

    assert tagger._input_name == "images"  # type: ignore[attr-defined]
    assert tagger._output_names == ["logits"]  # type: ignore[attr-defined]
    results = tagger.infer_batch([np.zeros((4, 4, 3), dtype=np.uint8)])

    assert len(results) == 1
    assert [prediction.name for prediction in results[0].tags] == ["tag_b", "tag_a"]
