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
    assert captured == dict(
        model="model.onnx", labels_json="custom.json", providers=["CUDAExecutionProvider"], max_batch=1
    )


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


class ProviderSession(Session):
    """Extend the fake with provider reporting for the real constructor."""

    def get_providers(self) -> list[str]:
        """Report the CPU provider like ORT does."""
        return ["CPUExecutionProvider"]


def _install_fake_ort(monkeypatch, factory) -> None:
    """Route the real constructor through a fake ORT module without the optional runtime."""
    from tagger import wd14_onnx

    fake = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=factory,
    )
    monkeypatch.setattr(wd14_onnx, "ensure_onnxruntime", lambda: None)
    monkeypatch.setattr(wd14_onnx, "ort", fake, raising=False)


@pytest.mark.parametrize("name", ["probs", "output", "sigmoid"])
def test_non_logits_output_name_is_rejected(name: str) -> None:
    """Same shape/dtype probabilities would be sigmoid-ed twice, so unverified names fail."""
    session = Session(1)
    session.get_outputs = lambda: [SimpleNamespace(name=name, shape=[1, LABEL_COUNT], type="tensor(float)")]
    with pytest.raises(ValueError, match="verified logits"):
        validate_session(session)


@pytest.mark.parametrize(
    "shape,dtype",
    [([1, LABEL_COUNT + 1], "tensor(float)"), ([1, LABEL_COUNT], "tensor(float16)"), ([LABEL_COUNT], "tensor(float)")],
)
def test_wrong_output_contract_is_rejected(shape, dtype) -> None:
    """Shape and dtype mismatches are rejected even when the name is logits."""
    session = Session(1)
    session.get_outputs = lambda: [SimpleNamespace(name="logits", shape=shape, type=dtype)]
    with pytest.raises(ValueError, match="float32 logits"):
        validate_session(session)


def test_constructor_starts_dynamic_batches_at_requested_size(monkeypatch, labels_file: Path) -> None:
    """The real constructor must not pin dynamic models to batch 1 before any OOM."""
    model = labels_file.with_name("model.onnx")
    model.write_bytes(b"graph")
    sessions: list[ProviderSession] = []

    def factory(*_args, **_kwargs) -> ProviderSession:
        sessions.append(ProviderSession(None))
        return sessions[-1]

    _install_fake_ort(monkeypatch, factory)
    tagger = PixaiV1OnnxTagger(model, providers=["CPUExecutionProvider"], max_batch=4)
    assert len(tagger.infer_batch_prepared(np.zeros((4, 3, 1008, 1008), np.float32))) == 4
    assert sessions[0].calls == [4]
    sessions[0].oom = True
    assert len(tagger.infer_batch_prepared(np.zeros((3, 3, 1008, 1008), np.float32))) == 3
    assert sessions[0].calls[1:] == [3, 1, 1, 1]


def test_constructor_rejects_files_replaced_while_loading(monkeypatch, labels_file: Path) -> None:
    """A model swapped during session creation must not be paired with the pre-load identity."""
    model = labels_file.with_name("model.onnx")
    model.write_bytes(b"graph-a")

    def factory(*_args, **_kwargs) -> ProviderSession:
        model.write_bytes(b"graph-b-longer")
        return ProviderSession(1)

    _install_fake_ort(monkeypatch, factory)
    with pytest.raises(RuntimeError, match="changed while loading"):
        PixaiV1OnnxTagger(model, providers=["CPUExecutionProvider"])


def test_explicit_labels_missing_or_corrupt_fail_signature(labels_file: Path) -> None:
    """An explicit bad path must not silently fall back to the adjacent tags.json identity."""
    model = labels_file.with_name("model.onnx")
    model.write_bytes(b"graph")
    settings = PipelineSettings(tagger=TaggerSettings(name="pixai-v1-onnx", model_path=str(model)))
    current_tagger_sig(settings)
    settings.tagger.labels_json = str(labels_file.with_name("custom.json"))
    with pytest.raises(ValueError, match="labels not found"):
        current_tagger_sig(settings)
    broken = labels_file.with_name("broken.json")
    broken.write_text(json.dumps(dict(num_classes=1)), encoding="utf-8")
    settings.tagger.labels_json = str(broken)
    with pytest.raises(ValueError, match="30877"):
        current_tagger_sig(settings)


def test_reference_summary_fails_on_threshold_flip() -> None:
    """allclose differences that cross a threshold change stored tags and must fail validation."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools" / "validate_pixai_v1.py"
    spec = importlib.util.spec_from_file_location("validate_pixai_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference, actual, threshold = 0.1699, 0.1701, 0.17
    flipped = dict(
        allclose=bool(np.allclose(actual, reference, atol=1e-3, rtol=1e-3)),
        threshold_disagreements=int((actual > threshold) != (reference > threshold)),
    )
    assert flipped == dict(allclose=True, threshold_disagreements=1)
    summary = module.summarize_reference([flipped])
    assert summary["numerical_pass"] and not summary["decision_pass"] and not summary["passed"]
    assert module.summarize_reference([dict(allclose=True, threshold_disagreements=0)])["passed"]


def test_preprocessing_avoids_full_size_float_copy() -> None:
    """Large sources must not allocate a 12 bytes/pixel float32 copy of the whole image."""
    import tracemalloc

    image = np.random.default_rng(0).integers(0, 256, (4000, 3000, 3), dtype=np.uint8)
    tracemalloc.start()
    try:
        prepare_rgb_batch([image])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < image.nbytes * 2  # a full float32 copy alone would be 4x


def test_source_pixel_guard(tmp_path: Path) -> None:
    """Oversized sources are refused before decoding instead of exhausting CPU memory."""
    from tagger.pixai_v1_preprocessing import SourceImageTooLargeError, load_source_rgb

    path = tmp_path / "big.png"
    Image.new("RGB", (300, 200)).save(path)
    assert load_source_rgb(path, max_pixels=60_000)[1] == (300, 200)
    with pytest.raises(SourceImageTooLargeError):
        load_source_rgb(path, max_pixels=59_999)


class _CaptureTagger:
    """Record prepared tensors from either inference path."""

    preserve_source_rgb = True
    preprocess_version = "capture"

    def __init__(self) -> None:
        self.prepared: list[np.ndarray] = []

    def prepare_batch_from_rgb_np(self, images):
        batch = prepare_rgb_batch(images)
        self.prepared.append(batch)
        return batch

    def infer_batch_prepared(self, batch, *, thresholds=None, max_tags=None):
        from tagger.base import TagResult

        return [TagResult([]) for _ in batch]

    def infer_batch(self, images, **_kwargs):
        raise AssertionError("v1 must not use the safe_load_image path")


def test_single_image_and_batch_paths_produce_identical_tensors(tmp_path: Path) -> None:
    """tag_job must not apply safe_load_image's 4096px shrink or EXIF rotation to v1 input."""
    from core.pipeline.loaders import PrefetchLoaderPrepared
    from core.tag_job import run_tag_job
    from db.connection import get_conn

    path = tmp_path / "wide.jpg"
    pixels = np.random.default_rng(1).integers(0, 256, (300, 5000, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW
    Image.fromarray(pixels).save(path, exif=exif.tobytes(), quality=95)

    batch_tagger = _CaptureTagger()
    loader = PrefetchLoaderPrepared([str(path)], tagger=batch_tagger, batch_size=1, prefetch_batches=1, io_workers=1)
    try:
        list(loader)
    finally:
        loader.close()
    single_tagger = _CaptureTagger()
    conn = get_conn(tmp_path / "db.sqlite")
    try:
        assert run_tag_job(single_tagger, path, conn) is not None  # type: ignore[arg-type]
    finally:
        conn.close()
    np.testing.assert_array_equal(batch_tagger.prepared[0], single_tagger.prepared[0])
    # 横長のまま（EXIF 回転なし）で 1008px に縮小され、上端は黒パディング
    assert np.all(single_tagger.prepared[0][0, :, 0] == -1)


def test_tag_stage_aborts_when_model_changed_after_signature(monkeypatch, labels_file: Path) -> None:
    """Results must not be written under a signature computed for different model content."""
    from core.pipeline.stages import tag_stage
    from core.pipeline.types import PipelineContext
    from tagger.pixai_v1_identity import settings_content_identity

    model = labels_file.with_name("model.onnx")
    model.write_bytes(b"graph-a")
    settings = PipelineSettings(tagger=TaggerSettings(name="pixai-v1-onnx", model_path=str(model)))
    signature = current_tagger_sig(settings)
    model.write_bytes(b"graph-b-longer")
    closed: list[int] = []
    loaded = SimpleNamespace(
        content_identity=settings_content_identity(settings.tagger), close=lambda: closed.append(1)
    )
    monkeypatch.setattr(tag_stage, "_resolve_tagger", lambda *_a, **_k: (loaded, None, None))
    ctx = PipelineContext(db_path="unused", settings=settings, thresholds={}, max_tags_map={}, tagger_sig=signature)
    record = SimpleNamespace(needs_tagging=True, load_failed=False)
    emitter = SimpleNamespace(emit=lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="changed after the tagger signature"):
        tag_stage.TagStage().run(ctx, emitter, [record])  # type: ignore[arg-type, list-item]
    assert closed == [1]
