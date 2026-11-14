"""Behavioral tests for PixAI v1 labels, preprocessing, inference and identity."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core.config import PipelineSettings, TaggerSettings
from core.pipeline.signature import current_tagger_sig
from tagger.base import TagCategory
from tagger.pixai_v1_labels import CATEGORY_COUNTS, LABEL_COUNT, load_labels_json, resolve_labels_json
from tagger.pixai_v1_onnx import PixaiV1OnnxTagger, validate_session
from tagger.pixai_v1_preprocessing import composite_rgb, prepare_rgb_batch


@pytest.fixture
def labels_file(tmp_path: Path) -> Path:
    """Create a complete six-category map with independently identifiable tags."""
    groups = []
    offset = 0
    for name, count in CATEGORY_COUNTS.items():
        groups.append(dict(name=name, count=count, offset=offset, tags=[f"{name}_{i}" for i in range(count)]))
        offset += count
    path = tmp_path / "tags.json"
    path.write_text(json.dumps(dict(num_classes=LABEL_COUNT, categories=groups)), encoding="utf-8")
    return path


def test_label_order_and_upstream_format(labels_file: Path) -> None:
    """Both formats must map output offsets to identical names and categories."""
    labels = load_labels_json(labels_file)
    assert labels[15043].name == "character_0"
    assert labels[25811].category == TagCategory.STYLE
    assert labels[-1].name == "rating_3"
    data = json.loads(labels_file.read_text())
    upstream = dict(
        num_classes=LABEL_COUNT,
        tags=[label.name for label in labels],
        tags_split=[[g["name"], g["count"]] for g in data["categories"]],
    )
    config = labels_file.with_name("config.json")
    config.write_text(json.dumps(upstream), encoding="utf-8")
    assert load_labels_json(config) == labels
    assert resolve_labels_json(labels_file.with_name("model.onnx")) == labels_file
    data["categories"][1]["offset"] = 0
    labels_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        load_labels_json(labels_file)


def test_preprocessing_keeps_edges_and_white_alpha() -> None:
    """Wide images keep both edges and use black padding, with RGB normalization."""
    image = np.full((2, 4, 3), [255, 0, 128], dtype=np.uint8)
    batch = prepare_rgb_batch([image])
    assert batch.shape == (1, 3, 1008, 1008)
    assert batch.dtype == np.float32
    assert np.all(batch[0, :, :252] == -1)
    np.testing.assert_allclose(batch[0, :, 252, 0], [1, -1, 128 / 127.5 - 1], atol=1e-6)
    np.testing.assert_array_equal(batch[0, :, 252, 0], batch[0, :, 755, -1])
    assert np.all(composite_rgb(Image.new("RGBA", (2, 2), (0, 0, 0, 0))) == 255)
    assert prepare_rgb_batch([]).shape == (0, 3, 1008, 1008)
    with pytest.raises(ValueError):
        prepare_rgb_batch([np.zeros((2, 2, 3), dtype=np.float32)])


class Session:
    """Predict sample identifiers to expose batch reordering and padding bugs."""

    def __init__(self, fixed: int | None = 2) -> None:
        self.fixed = fixed
        self.calls: list[int] = []
        self.oom = False

    def get_inputs(self) -> list[SimpleNamespace]:
        """Return the supported input contract."""
        return [SimpleNamespace(name="pixels", shape=[self.fixed, 3, 1008, 1008], type="tensor(float)")]

    def get_outputs(self) -> list[SimpleNamespace]:
        """Return ordered logits metadata."""
        return [SimpleNamespace(name="logits", shape=[self.fixed, LABEL_COUNT], type="tensor(float)")]

    def run(self, outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Emit logits, optionally simulating an allocation failure."""
        values = inputs["pixels"]
        self.calls.append(len(values))
        if self.oom and len(values) > 1:
            raise RuntimeError("CUDA out of memory")
        logits = np.full((len(values), LABEL_COUNT), -1000, np.float32)
        for i, value in enumerate(values):
            logits[i, int(value[0, 0, 0])] = 1000
            logits[i, 25811] = 0  # style probability .5
        return [logits]


def make_tagger(labels_file: Path, fixed: int | None = 2) -> PixaiV1OnnxTagger:
    """Build a backend using a deterministic session without optional ORT."""
    tagger = object.__new__(PixaiV1OnnxTagger)
    tagger._session = Session(fixed)
    tagger._labels = load_labels_json(labels_file)
    tagger._input_name, tagger._output_name, tagger._fixed_batch = validate_session(tagger._session)
    tagger._batch_limit = fixed or 4
    return tagger


def test_fixed_batch_tail_and_six_category_filter(labels_file: Path) -> None:
    """Fixed batching discards padding and applies v1 style thresholds exactly once."""
    tagger = make_tagger(labels_file)
    batch = np.stack([np.full((3, 1008, 1008), i, np.float32) for i in range(3)])
    results = tagger.infer_batch_prepared(batch, thresholds={TagCategory.STYLE: 0.6})
    assert [[p.name for p in result.tags] for result in results] == [[f"general_{i}"] for i in range(3)]
    assert tagger._session.calls == [2, 2]
    results = tagger.infer_batch_prepared(batch[:1])
    assert any(p.category == TagCategory.STYLE and p.score == 0.5 for p in results[0].tags)
    batch[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tagger.infer_batch_prepared(batch)


def test_dynamic_batch_oom_retry(labels_file: Path) -> None:
    """OOM retries retain order and permanently lower the dynamic batch limit."""
    tagger = make_tagger(labels_file, None)
    tagger._session.oom = True
    batch = np.zeros((3, 3, 1008, 1008), np.float32)
    assert len(tagger.infer_batch_prepared(batch)) == 3
    assert tagger._session.calls == [3, 1, 1, 1]


def test_v1_settings_signature_changes_with_content(labels_file: Path) -> None:
    """Defaults survive serialization and replacing model/weights invalidates identity."""
    model = labels_file.with_name("model.onnx")
    model.write_bytes(b"original")
    settings = PipelineSettings(tagger=TaggerSettings(name="pixai-v1-onnx", model_path=str(model)))
    assert settings.batch_size == 1 and settings.prefetch_depth == 1
    assert settings.tagger.thresholds["general"] == 0.17
    restored = PipelineSettings.model_validate(settings.model_dump())
    before = current_tagger_sig(restored)
    model.write_bytes(b"replaced")
    assert current_tagger_sig(restored) != before
    before = current_tagger_sig(restored)
    model.with_name("model.onnx.data").write_bytes(b"weights")
    assert current_tagger_sig(restored) != before
    before = current_tagger_sig(restored)
    restored.tagger.thresholds["style"] = 0.7
    assert current_tagger_sig(restored) != before


def test_inspection_reports_v1_contract(labels_file: Path) -> None:
    """Diagnostics validates v1 labels and rejects old-size tensors."""
    from tagger.model_inspection import format_inspection, inspect_model

    model = labels_file.with_name("model.onnx")
    model.write_bytes(b"test")
    inspection = inspect_model(
        tagger_name="pixai-v1-onnx",
        model_path=model,
        tags_csv=None,
        provider_loader=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: Session(1),
    )
    assert inspection.ok
    assert inspection.category_counts["style"] == 4917
    assert "Labels JSON:" in format_inspection(inspection)
    assert inspection.output_dim == LABEL_COUNT
    labels_file.unlink()
    inspection = inspect_model(
        tagger_name="pixai-v1-onnx",
        model_path=model,
        tags_csv=None,
        provider_loader=lambda: [],
        session_factory=lambda *_: Session(1),
    )
    assert not inspection.ok


def test_resolver_selects_v1_and_keeps_explicit_labels(monkeypatch, tmp_path: Path) -> None:
    """The pipeline routes v1 to its own backend, independently of old provider settings."""
    from core.pipeline.resolver import _resolve_tagger
    from tagger import pixai_v1_onnx

    captured = {}

    def create(model_path, **kwargs):
        captured.update(model=model_path, **kwargs)
        return SimpleNamespace(provider="pixai-v1")

    monkeypatch.setattr(pixai_v1_onnx, "PixaiV1OnnxTagger", create)
    settings = PipelineSettings(
        tagger=TaggerSettings(
            name="pixai-v1-onnx", model_path="model.onnx", labels_json="custom.json", provider="wd14", device="cuda"
        )
    )
    tagger, _, _ = _resolve_tagger(settings, None)
    assert tagger.provider == "pixai-v1"
    assert captured == dict(model="model.onnx", labels_json="custom.json", providers=["CUDAExecutionProvider"])


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ([1, 3, 448, 448], "tensor(float)"),
        ([1, 1008, 1008, 3], "tensor(float)"),
        ([1, 3, 1008, 1008], "tensor(float16)"),
    ],
)
def test_wrong_input_contract_is_rejected(shape, dtype) -> None:
    """Old preprocessing and half precision models cannot silently use the v1 path."""
    session = Session(1)
    session.get_inputs = lambda: [SimpleNamespace(name="pixels", shape=shape, type=dtype)]
    with pytest.raises(ValueError, match="float32 NCHW"):
        validate_session(session)


def test_signature_uses_effective_defaults(labels_file: Path) -> None:
    """Partial overrides must have the same identity as their expanded effective settings."""
    model = labels_file.with_name("model.onnx")
    model.write_bytes(b"graph")
    settings = PipelineSettings(tagger=TaggerSettings(name="pixai-v1-onnx", model_path=str(model)))
    assert current_tagger_sig(settings, thresholds={TagCategory.GENERAL: 0.17}) == current_tagger_sig(settings)
    settings.tagger.labels_json = str(labels_file)
    before = current_tagger_sig(settings)
    data = json.loads(labels_file.read_text())
    data["categories"][0]["tags"][0] = "changed_tag"
    labels_file.write_text(json.dumps(data), encoding="utf-8")
    assert current_tagger_sig(settings) != before
