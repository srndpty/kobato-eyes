"""Tests for index pipeline stage overrides and cancellation behaviour."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("KOE_HEADLESS", "1")


if "yaml" not in sys.modules:
    fake_yaml = types.ModuleType("yaml")

    class _YamlError(Exception):
        pass

    def _safe_load(_text):
        return {}

    def _safe_dump(_data, sort_keys=False):
        return ""

    fake_yaml.safe_load = _safe_load
    fake_yaml.safe_dump = _safe_dump
    fake_yaml.YAMLError = _YamlError
    sys.modules["yaml"] = fake_yaml


if "pydantic" not in sys.modules:
    def _field(default=None, *, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    def _copy_default(value):
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, set):
            return set(value)
        return value

    class _FakeBaseModel:
        model_config: dict[str, object] = {}

        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in kwargs:
                    continue
                default = getattr(self.__class__, name, None)
                setattr(self, name, _copy_default(default))
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data):
            if not isinstance(data, dict):
                data = {}
            return cls(**data)

        def model_dump(self):
            annotations = getattr(self.__class__, "__annotations__", {})
            return {name: getattr(self, name) for name in annotations}

    def _pass_through_decorator(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = _FakeBaseModel
    fake_pydantic.ConfigDict = lambda **kwargs: dict(kwargs)
    fake_pydantic.Field = _field
    fake_pydantic.field_validator = _pass_through_decorator
    fake_pydantic.model_validator = _pass_through_decorator
    sys.modules["pydantic"] = fake_pydantic


if "numpy" not in sys.modules:
    class _FakeNdArray(list):
        pass

    def _array(data, dtype=None):  # noqa: ARG001 - dtype kept for compatibility
        if isinstance(data, _FakeNdArray):
            return _FakeNdArray(data)
        return _FakeNdArray(data)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = _FakeNdArray
    fake_numpy.array = _array
    sys.modules["numpy"] = fake_numpy


if "PIL" not in sys.modules:
    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_imagefile = types.ModuleType("PIL.ImageFile")

    class _FakeImage:  # noqa: D401 - simple placeholder
        """Minimal stand-in for ``PIL.Image.Image``."""

        def __init__(self, *args, **kwargs):  # noqa: D401, ANN002 - compatibility only
            self.args = args
            self.kwargs = kwargs

    fake_pil_image.Image = _FakeImage
    fake_pil_image.DecompressionBombError = type(
        "DecompressionBombError",
        (RuntimeError,),
        {},
    )
    fake_pil.Image = fake_pil_image
    fake_pil_imagefile.ImageFile = _FakeImage
    fake_pil.ImageFile = fake_pil_imagefile
    fake_pil.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})
    sys.modules["PIL"] = fake_pil
    sys.modules["PIL.Image"] = fake_pil_image
    sys.modules["PIL.ImageFile"] = fake_pil_imagefile


if "cv2" not in sys.modules:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_GRAY2BGR = 0
    fake_cv2.COLOR_BGR2RGB = 1
    fake_cv2.INTER_AREA = 0
    fake_cv2.INTER_CUBIC = 1
    fake_cv2.IMREAD_UNCHANGED = -1

    def _cv_identity(*args, **kwargs):  # noqa: D401, ANN002 - compatibility only
        return args[0] if args else None

    def _cv_imdecode(*_args, **_kwargs):
        return None

    fake_cv2.cvtColor = _cv_identity
    fake_cv2.resize = _cv_identity
    fake_cv2.imdecode = _cv_imdecode
    sys.modules["cv2"] = fake_cv2


from core.pipeline.orchestrator import IndexPipeline
from core.pipeline.stages.scan_stage import ScanStageResult
from core.pipeline.stages.tag_stage import TagStageResult
from core.pipeline.stages.write_stage import WriteStageResult
from core.pipeline.types import PipelineContext, _FileRecord


class FakeScanStage:
    def __init__(self) -> None:
        self.called = False

    def run(self, ctx: PipelineContext, emitter, *_):
        self.called = True
        record = _FileRecord(
            file_id=1,
            path=Path("/tmp/a.jpg"),
            size=1,
            mtime=0.0,
            sha="x",
            is_new=True,
            changed=False,
            tag_exists=False,
            needs_tagging=True,
        )
        return ScanStageResult(records=[record], scanned=1, new_or_changed=1)


class FakeTagStage:
    def __init__(self) -> None:
        self.called = False
        self.received: list[_FileRecord] | None = None

    def run(self, ctx: PipelineContext, emitter, records):
        self.called = True
        self.received = records
        return TagStageResult(records=records, db_items=[], tagged_count=5)


class FakeWriteStage:
    def __init__(self) -> None:
        self.called = False
        self.received = None

    def run(self, ctx: PipelineContext, emitter, tag_result):
        self.called = True
        self.received = tag_result
        return WriteStageResult(written=3, fts_processed=3)


def test_index_pipeline_allows_stage_overrides(tmp_path):
    pipeline = IndexPipeline(db_path=tmp_path / "test.db")
    fake_scan = FakeScanStage()
    fake_tag = FakeTagStage()
    fake_write = FakeWriteStage()

    pipeline.set_stage_override("scan", fake_scan)
    pipeline.set_stage_override("tag", fake_tag)
    pipeline.set_stage_override("write", fake_write)

    stats = pipeline.run()

    assert fake_scan.called
    assert fake_tag.received is not None
    assert fake_write.received is not None
    assert stats["scanned"] == 1
    assert stats["new_or_changed"] == 1
    assert stats["tagged"] == 5
    assert stats["signatures"] == 3


def test_index_pipeline_stops_on_cancellation(tmp_path):
    cancelled = False

    def is_cancelled() -> bool:
        nonlocal cancelled
        if not cancelled:
            cancelled = True
        return cancelled

    pipeline = IndexPipeline(db_path=tmp_path / "test.db", is_cancelled=is_cancelled)
    fake_scan = FakeScanStage()
    fake_tag = FakeTagStage()
    fake_write = FakeWriteStage()

    pipeline.set_stage_override("scan", fake_scan)
    pipeline.set_stage_override("tag", fake_tag)
    pipeline.set_stage_override("write", fake_write)

    stats = pipeline.run()

    assert fake_scan.called
    assert not fake_tag.called
    assert not fake_write.called
    assert stats["scanned"] == 1
    assert stats["new_or_changed"] == 1
    assert stats["tagged"] == 0
    assert stats["signatures"] == 0
    assert stats["cancelled"] is True
