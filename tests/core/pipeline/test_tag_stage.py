"""Tests for TagStage's prepared batch retry logic."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Iterable, Sequence


class _FakeClipResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def astype(self, _dtype: object) -> object:
        return self._value


_numpy_stub = types.ModuleType("numpy")
_numpy_stub.ndarray = object
_numpy_stub.float32 = "float32"
_numpy_stub.uint8 = "uint8"
_numpy_stub.clip = lambda array, _min, _max: _FakeClipResult(array)
sys.modules.setdefault("numpy", _numpy_stub)

_config_stub = types.ModuleType("core.config")
_config_stub.load_settings = lambda *args, **kwargs: None
_config_stub.AppPaths = type("AppPaths", (), {})
_config_stub.PipelineSettings = type("PipelineSettings", (), {})
sys.modules.setdefault("core.config", _config_stub)

_core_stub = types.ModuleType("core")
_repo_root = Path(__file__).resolve().parents[3]

_core_stub.__path__ = [str(_repo_root / "src" / "core")]
_core_stub.__spec__ = importlib.machinery.ModuleSpec("core", loader=None, is_package=True)
sys.modules.setdefault("core", _core_stub)

_core_pipeline_stub = types.ModuleType("core.pipeline")
_core_pipeline_stub.__path__ = [str(_repo_root / "src" / "core" / "pipeline")]
_core_pipeline_stub.__spec__ = importlib.machinery.ModuleSpec("core.pipeline", loader=None, is_package=True)
sys.modules.setdefault("core.pipeline", _core_pipeline_stub)
sys.modules.pop("core.pipeline.stages", None)

_core_pipeline_resolver_stub = types.ModuleType("core.pipeline.resolver")


def _resolver_stub(settings, override, *, thresholds=None, max_tags=None):
    if override is None:
        raise AssertionError("tagger override must be provided for tests")
    return override


_pil_stub = types.ModuleType("PIL")
_pil_image_stub = types.ModuleType("PIL.Image")
_pil_image_stub.fromarray = lambda array: array
_pil_image_stub.DecompressionBombError = type("DecompressionBombError", (Exception,), {})
_pil_imagefile_stub = types.ModuleType("PIL.ImageFile")
_pil_imagefile_stub.LOAD_TRUNCATED_IMAGES = False
_pil_stub.Image = _pil_image_stub
_pil_stub.ImageFile = _pil_imagefile_stub
_pil_stub.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})
sys.modules.setdefault("PIL", _pil_stub)
sys.modules.setdefault("PIL.Image", _pil_image_stub)
sys.modules.setdefault("PIL.ImageFile", _pil_imagefile_stub)

_core_pipeline_resolver_stub._resolve_tagger = _resolver_stub
sys.modules.setdefault("core.pipeline.resolver", _core_pipeline_resolver_stub)

_core_pipeline_stages_stub = types.ModuleType("core.pipeline.stages")
_core_pipeline_stages_stub.__path__ = [str(_repo_root / "src" / "core" / "pipeline" / "stages")]
_core_pipeline_stages_stub.__spec__ = importlib.machinery.ModuleSpec(
    "core.pipeline.stages", loader=None, is_package=True
)
sys.modules.setdefault("core.pipeline.stages", _core_pipeline_stages_stub)

_tag_stage_path = _repo_root / "src" / "core" / "pipeline" / "stages" / "tag_stage.py"
_tag_stage_spec = importlib.util.spec_from_file_location(
    "core.pipeline.stages.tag_stage",
    _tag_stage_path,
)
assert _tag_stage_spec and _tag_stage_spec.loader
_tag_stage_module = importlib.util.module_from_spec(_tag_stage_spec)
sys.modules.setdefault("core.pipeline.stages.tag_stage", _tag_stage_module)
_tag_stage_spec.loader.exec_module(_tag_stage_module)

TagStage = _tag_stage_module.TagStage
TagStageDeps = _tag_stage_module.TagStageDeps
from core.pipeline.types import IndexPhase, PipelineContext, ProgressEmitter, _FileRecord
from tagger.base import ITagger, TagCategory, TagPrediction, TagResult


class _FakeSample:
    """Simple placeholder for prepared RGB arrays."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)


class _FakeBatch:
    """Minimal batch object supporting slicing and iteration."""

    def __init__(self, samples: Sequence[_FakeSample]) -> None:
        self._samples = list(samples)

    def __iter__(self):
        return iter(self._samples)

    def __len__(self) -> int:  # pragma: no cover - len() calls
        return len(self._samples)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _FakeBatch(self._samples[key])
        return self._samples[key]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        width = self._samples[0].width if self._samples else 0
        height = self._samples[0].height if self._samples else 0
        return (len(self._samples), height, width, 3)


class _RetryingLoader:
    """Loader that yields a single batch covering all provided paths."""

    def __init__(self, paths: list[str]) -> None:
        self._paths = paths
        self._yielded = False
        self.closed = False

    def __iter__(self) -> Iterable[tuple[list[str], _FakeBatch, list[tuple[int, int]]]]:
        if self._yielded:
            return iter(())
        self._yielded = True
        samples = [_FakeSample(64, 64) for _ in self._paths]
        sizes = [(64, 64)] * len(self._paths)
        yield (self._paths, _FakeBatch(samples), sizes)

    def close(self) -> None:
        self.closed = True


class _RetryingDeps(TagStageDeps):
    """Dependency provider that returns the retrying loader."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def loader_factory(
        self,
        paths: list[str],
        tagger,
        batch_size: int,
        prefetch_batches: int,
        io_workers: int | None,
    ) -> _RetryingLoader:
        self.batch_sizes.append(batch_size)
        return _RetryingLoader(list(paths))


class _FlakyTagger(ITagger):
    """Tagger that fails for batches larger than one element."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def prepare_batch_from_rgb_np(self, images: Sequence[object]):
        return _FakeBatch([_FakeSample(64, 64) for _ in images])

    def infer_batch_prepared(
        self,
        batch: _FakeBatch,
        *,
        thresholds=None,
        max_tags=None,
    ) -> list[TagResult]:
        size = len(batch)
        self.batch_sizes.append(size)
        if size > 1:
            raise RuntimeError("prepared batch too large")
        return [
            TagResult(tags=[TagPrediction(name="retry", score=0.99, category=TagCategory.GENERAL)]) for _ in range(size)
        ]

    def infer_batch(self, images, *, thresholds=None, max_tags=None):
        raise AssertionError("infer_batch should not be used when infer_batch_prepared is available")


class _DummyTaggerSettings:
    def __init__(self) -> None:
        self.name = "wd14-onnx"
        self.model_path = "dummy.onnx"
        self.provider = "wd14"  # ← 任意
        self.tags_csv = None  # ← これを追加


class _DummySettings:
    def __init__(self, *, batch_size: int = 8) -> None:
        self.tagger = _DummyTaggerSettings()
        self.batch_size = batch_size


def _make_context(tagger: ITagger, *, batch_size: int = 8) -> PipelineContext:
    settings = _DummySettings(batch_size=batch_size)
    return PipelineContext(
        db_path=":memory:",
        settings=settings,
        thresholds={TagCategory.GENERAL: 0.5},
        max_tags_map={},
        tagger_sig="test:tagger",
        tagger_override=tagger,
        progress_cb=None,
        is_cancelled=None,
    )


def test_tag_stage_retries_failed_prepared_batches() -> None:
    records = [
        _FileRecord(
            file_id=i,
            path=Path(f"file_{i}.jpg"),
            size=1,
            mtime=time.time(),
            sha=f"sha{i}",
            is_new=True,
            changed=False,
            tag_exists=False,
            needs_tagging=True,
        )
        for i in range(4)
    ]
    progress_events: list[tuple[IndexPhase, int, int]] = []
    emitter = ProgressEmitter(lambda p: progress_events.append((p.phase, p.done, p.total)))

    tagger = _FlakyTagger()
    stage = TagStage(deps=_RetryingDeps())
    ctx = _make_context(tagger)

    result = stage.run(ctx, emitter, records)

    assert result.tagged_count == len(records)
    assert len(result.db_items) == len(records)
    assert all(not rec.needs_tagging for rec in result.records)
    assert tagger.batch_sizes and tagger.batch_sizes[0] == len(records)
    assert any(size == 1 for size in tagger.batch_sizes)
    assert len(tagger.batch_sizes) > len(records)
    assert progress_events[0][0] == IndexPhase.TAG
    assert progress_events[-1][1] == len(records)
    assert progress_events[-1][2] == len(records)


def test_tag_stage_uses_configured_batch_size() -> None:
    records = [
        _FileRecord(
            file_id=1,
            path=Path("single.jpg"),
            size=1,
            mtime=time.time(),
            sha="sha1",
            is_new=True,
            changed=False,
            tag_exists=False,
            needs_tagging=True,
        )
    ]
    emitter = ProgressEmitter(lambda _p: None)

    deps = _RetryingDeps()
    tagger = _FlakyTagger()
    stage = TagStage(deps=deps)
    ctx = _make_context(tagger, batch_size=17)

    stage.run(ctx, emitter, records)

    assert deps.batch_sizes and deps.batch_sizes[0] == 17

    # 1回目の実行で records[0].needs_tagging が False になっているため、
    # 2回目は新しいレコードを作り直す
    records = [
        _FileRecord(
            file_id=1,
            path=Path("single.jpg"),
            size=1,
            mtime=time.time(),
            sha="sha1",
            is_new=True,
            changed=False,
            tag_exists=False,
            needs_tagging=True,
        )
    ]
    deps = _RetryingDeps()
    tagger = _FlakyTagger()
    stage = TagStage(deps=deps)
    ctx = _make_context(tagger, batch_size=0)
    stage.run(ctx, emitter, records)
    assert deps.batch_sizes and deps.batch_sizes[0] == 1
