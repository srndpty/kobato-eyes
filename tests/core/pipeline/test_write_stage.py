"""WriteStage におけるエラー時クリーンアップの挙動を検証するテスト。"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

os.environ.setdefault("KOE_HEADLESS", "1")


if "yaml" not in sys.modules:
    fake_yaml = types.ModuleType("yaml")

    class _YamlError(Exception):
        pass

    fake_yaml.safe_load = lambda _text: {}
    fake_yaml.safe_dump = lambda _data, sort_keys=False: ""
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

    class _FakeImage:
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN002 - 互換性確保のダミー
            self.args = args
            self.kwargs = kwargs

    fake_pil_image.Image = _FakeImage
    fake_pil_image.DecompressionBombError = type("DecompressionBombError", (RuntimeError,), {})
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
    fake_cv2.cvtColor = lambda *args, **kwargs: args[0] if args else None
    fake_cv2.resize = lambda *args, **kwargs: args[0] if args else None
    fake_cv2.imdecode = lambda *_args, **_kwargs: None
    sys.modules["cv2"] = fake_cv2


from core.pipeline.stages import write_stage
from core.pipeline.stages.tag_stage import TagStageResult
from core.pipeline.stages.write_stage import WriteStage
from core.pipeline.types import PipelineContext, ProgressEmitter


def _make_ctx(tmp_path) -> PipelineContext:
    settings = SimpleNamespace(db_flush_chunk=16, fts_topk=32)
    return PipelineContext(
        db_path=tmp_path / "test.db",
        settings=settings,
        thresholds={},
        max_tags_map={},
        tagger_sig="sig",
        tagger_override=None,
        progress_cb=None,
        is_cancelled=lambda: False,
    )


def test_write_stage_cleans_up_after_writer_failure(monkeypatch, tmp_path):
    settle_calls: list[str] = []

    monkeypatch.setattr(
        write_stage,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(str(path)),
    )

    class _FailingWriter:
        def __init__(self) -> None:
            self.stop_calls = 0

        def start(self) -> None:
            raise RuntimeError("writer failed during start")

        def raise_if_failed(self) -> None:  # pragma: no cover - never reached
            pytest.fail("raise_if_failed should not be called after start failure")

        def put(self, _item) -> None:  # pragma: no cover - never reached
            pytest.fail("put should not be called when start fails")

        def stop(self, *, flush: bool, wait_forever: bool) -> None:
            assert flush is True
            assert wait_forever is True
            self.stop_calls += 1

    class _Deps:
        def __init__(self) -> None:
            self.begin_calls = 0
            self.end_calls = 0
            self.writer = _FailingWriter()

        def connect(self, db_path: str) -> None:
            assert db_path

        def begin_quiesce(self) -> None:
            self.begin_calls += 1

        def end_quiesce(self) -> None:
            self.end_calls += 1

        def build_writer(self, *, ctx: PipelineContext, progress_cb):
            assert ctx
            assert callable(progress_cb)
            return self.writer

    deps = _Deps()
    stage = WriteStage(deps=deps)
    ctx = _make_ctx(tmp_path)
    emitter = ProgressEmitter(lambda _progress: None)
    tag_result = TagStageResult(records=[], db_items=[object()], tagged_count=1)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 0
    assert result.fts_processed == 0
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert deps.writer.stop_calls == 1
    assert settle_calls == [str(ctx.db_path)]


def test_write_stage_cleans_up_when_build_writer_fails(monkeypatch, tmp_path):
    settle_calls: list[str] = []

    monkeypatch.setattr(
        write_stage,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(str(path)),
    )

    class _Deps:
        def __init__(self) -> None:
            self.begin_calls = 0
            self.end_calls = 0
            self.connect_calls = 0

        def connect(self, db_path: str) -> None:
            assert db_path
            self.connect_calls += 1

        def begin_quiesce(self) -> None:
            self.begin_calls += 1

        def end_quiesce(self) -> None:
            self.end_calls += 1

        def build_writer(self, *, ctx: PipelineContext, progress_cb):
            assert ctx
            assert callable(progress_cb)
            raise RuntimeError("failed to build writer")

    deps = _Deps()
    stage = WriteStage(deps=deps)
    ctx = _make_ctx(tmp_path)
    emitter = ProgressEmitter(lambda _progress: None)
    tag_result = TagStageResult(records=[], db_items=[object()], tagged_count=1)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 0
    assert result.fts_processed == 0
    assert deps.connect_calls == 1
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert settle_calls == [str(ctx.db_path)]
