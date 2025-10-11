"""Tests for :mod:`core.pipeline.stages.write_stage`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List

import pytest


def _install_stub_modules() -> None:
    """Inject lightweight stand-ins for optional dependencies."""

    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")

        class YAMLError(Exception):
            pass

        def safe_load(_text):
            return {}

        def safe_dump(_payload, *, sort_keys: bool = False):
            return ""

        yaml_stub.YAMLError = YAMLError
        yaml_stub.safe_load = safe_load
        yaml_stub.safe_dump = safe_dump
        sys.modules["yaml"] = yaml_stub

    if "pydantic" not in sys.modules:
        pydantic_stub = types.ModuleType("pydantic")

        class _BaseModel:
            def __init__(self, **kwargs) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def model_dump(self) -> dict:
                return dict(self.__dict__)

            @classmethod
            def model_validate(cls, data):
                if not isinstance(data, dict):
                    data = {}
                return cls(**data)

        def _config_dict(**kwargs):
            return kwargs

        def _field(*, default=None, default_factory=None, **_kwargs):
            if default_factory is not None:
                return default_factory()
            return default

        def _decorator(*_args, **_kwargs):
            def _wrap(fn):
                return fn

            return _wrap

        pydantic_stub.BaseModel = _BaseModel
        pydantic_stub.ConfigDict = _config_dict
        pydantic_stub.Field = _field
        pydantic_stub.field_validator = _decorator
        pydantic_stub.model_validator = _decorator
        sys.modules["pydantic"] = pydantic_stub

    if "core.config" not in sys.modules:
        config_module = types.ModuleType("core.config")
        config_module.__path__ = []  # type: ignore[attr-defined]

        @dataclass
        class _TaggerSettings:
            name: str = "dummy"

        @dataclass
        class _PipelineSettings:
            db_flush_chunk: int = 1024
            fts_topk: int = 128
            tagger: _TaggerSettings = field(default_factory=_TaggerSettings)

            def resolved_index_dir(self, default: str) -> str:
                return default

            def to_mapping(self, *, default_index_dir: str | None = None) -> dict:
                return {}

            @classmethod
            def from_mapping(cls, data):
                return cls()

        class _AppPaths:
            def config_path(self, filename: str):
                return Path(filename)

            def index_dir(self):
                return Path("./index")

        class _SettingsService:
            def __init__(self, *_args, **_kwargs) -> None:
                self._paths = _AppPaths()

            @property
            def config_path(self):
                return Path("./config.yaml")

            def load(self):
                return _PipelineSettings()

            def save(self, _settings):
                return None

        def _configure(_app_paths):
            return None

        def _config_path() -> Path:
            return Path("./config.yaml")

        def _load_settings() -> _PipelineSettings:
            return _PipelineSettings()

        def _save_settings(_settings: _PipelineSettings) -> None:
            return None

        config_module.TaggerSettings = _TaggerSettings
        config_module.PipelineSettings = _PipelineSettings
        config_module.AppPaths = _AppPaths
        config_module.SettingsService = _SettingsService
        config_module.configure = _configure
        config_module.config_path = _config_path
        config_module.load_settings = _load_settings
        config_module.save_settings = _save_settings
        sys.modules["core.config"] = config_module

    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = object
        numpy_stub.float32 = float
        numpy_stub.uint8 = int
        sys.modules["numpy"] = numpy_stub

    if "PIL" not in sys.modules:
        pil_stub = types.ModuleType("PIL")
        pil_stub.__path__ = []  # type: ignore[attr-defined]
        image_module = types.ModuleType("PIL.Image")
        imagefile_module = types.ModuleType("PIL.ImageFile")

        class _Image:  # pragma: no cover - dummy placeholder
            pass

        class _ImageFile:  # pragma: no cover - dummy placeholder
            pass

        class _UnidentifiedImageError(Exception):
            pass

        class _DecompressionBombError(Exception):
            pass

        image_module.Image = _Image
        image_module.DecompressionBombError = _DecompressionBombError
        imagefile_module.ImageFile = _ImageFile
        pil_stub.Image = _Image
        pil_stub.ImageFile = _ImageFile
        pil_stub.UnidentifiedImageError = _UnidentifiedImageError
        pil_stub.DecompressionBombError = _DecompressionBombError
        sys.modules["PIL"] = pil_stub
        sys.modules["PIL.Image"] = image_module
        sys.modules["PIL.ImageFile"] = imagefile_module

    if "tagger" not in sys.modules:
        tagger_pkg = types.ModuleType("tagger")
        tagger_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["tagger"] = tagger_pkg

    if "tagger.base" not in sys.modules:
        from enum import Enum

        tagger_base = types.ModuleType("tagger.base")

        class TagCategory(Enum):
            GENERAL = 0
            CHARACTER = 1
            RATING = 2
            COPYRIGHT = 3
            ARTIST = 4
            META = 5

        class ITagger:  # pragma: no cover - protocol stand-in
            pass

        @dataclass(frozen=True)
        class TagPrediction:
            name: str
            score: float
            category: TagCategory

        @dataclass
        class TagResult:
            tags: list[TagPrediction]

        TagResult.__module__ = "tagger.base"  # type: ignore[attr-defined]

        tagger_base.TagCategory = TagCategory
        tagger_base.ITagger = ITagger
        tagger_base.TagPrediction = TagPrediction
        tagger_base.TagResult = TagResult
        tagger_base.ThresholdMap = dict
        tagger_base.MaxTagsMap = dict
        sys.modules["tagger.base"] = tagger_base
        sys.modules["tagger"].base = tagger_base  # type: ignore[index]

    if "services" not in sys.modules:
        services_pkg = types.ModuleType("services")
        services_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["services"] = services_pkg

    if "services.db_writing" not in sys.modules:
        db_writing_module = types.ModuleType("services.db_writing")

        class DBWritingService:  # pragma: no cover - minimal stub
            def __init__(self, *args, **kwargs) -> None:
                pass

            def start(self) -> None:
                return None

            def raise_if_failed(self) -> None:
                return None

            def put(self, item, block: bool = True, timeout: float | None = None) -> None:
                return None

            def qsize(self) -> int:
                return 0

            def stop(self, *, flush: bool = True, wait_forever: bool = False) -> None:
                return None

        db_writing_module.DBWritingService = DBWritingService
        sys.modules["services.db_writing"] = db_writing_module
        sys.modules["services"].db_writing = db_writing_module  # type: ignore[index]

    if "db" not in sys.modules:
        db_pkg = types.ModuleType("db")
        db_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["db"] = db_pkg

    if "PyQt6" not in sys.modules:
        pyqt_pkg = types.ModuleType("PyQt6")
        pyqt_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["PyQt6"] = pyqt_pkg

    if "PyQt6.QtCore" not in sys.modules:
        qtcore_module = types.ModuleType("PyQt6.QtCore")

        class QObject:  # pragma: no cover - stub
            def __init__(self, *args, **kwargs) -> None:
                return None

            def deleteLater(self) -> None:
                return None

        class QRunnable:  # pragma: no cover - stub
            def run(self) -> None:
                return None

        class QThreadPool:  # pragma: no cover - stub
            @staticmethod
            def globalInstance():
                return QThreadPool()

            def start(self, runnable) -> None:
                if hasattr(runnable, "run"):
                    runnable.run()

        class QTimer:  # pragma: no cover - stub
            def __init__(self, *args, **kwargs) -> None:
                self._interval = 0

            def start(self, interval: int) -> None:
                self._interval = interval

            def stop(self) -> None:
                self._interval = 0

        def pyqtSignal(*_args, **_kwargs):  # pragma: no cover - stub
            class _Signal:
                def connect(self, *_args, **_kwargs) -> None:
                    return None

                def emit(self, *_args, **_kwargs) -> None:
                    return None

            return _Signal()

        qtcore_module.QObject = QObject
        qtcore_module.QRunnable = QRunnable
        qtcore_module.QThreadPool = QThreadPool
        qtcore_module.QTimer = QTimer
        qtcore_module.pyqtSignal = pyqtSignal
        sys.modules["PyQt6.QtCore"] = qtcore_module
        sys.modules["PyQt6"].QtCore = qtcore_module  # type: ignore[index]

    for name in ("connection", "files", "fts", "tags", "repository"):
        module_name = f"db.{name}"
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        if name == "connection":

            def begin_quiesce() -> None:  # pragma: no cover - stub
                return None

            def end_quiesce() -> None:  # pragma: no cover - stub
                return None

            class _Conn:
                def close(self) -> None:  # pragma: no cover - stub
                    return None

            def get_conn(db_path: str, allow_when_quiesced: bool = False):  # pragma: no cover - stub
                return _Conn()

            def bootstrap_if_needed(_path: str) -> None:  # pragma: no cover - stub
                return None

            module.begin_quiesce = begin_quiesce  # type: ignore[attr-defined]
            module.end_quiesce = end_quiesce  # type: ignore[attr-defined]
            module.get_conn = get_conn  # type: ignore[attr-defined]
            module.bootstrap_if_needed = bootstrap_if_needed  # type: ignore[attr-defined]
        elif name == "files":

            def bulk_update_files_meta_by_id(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            module.bulk_update_files_meta_by_id = bulk_update_files_meta_by_id  # type: ignore[attr-defined]
        elif name == "fts":

            def fts_replace_rows(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            module.fts_replace_rows = fts_replace_rows  # type: ignore[attr-defined]
        elif name == "tags":

            def upsert_tags(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            module.upsert_tags = upsert_tags  # type: ignore[attr-defined]
        elif name == "repository":

            def replace_file_tags(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            def update_fts(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            def upsert_file(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            def upsert_tags_repo(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            def get_file_by_path(*_args, **_kwargs):  # pragma: no cover - stub
                return None

            def list_untagged_under_path(*_args, **_kwargs):  # pragma: no cover - stub
                return []

            module.replace_file_tags = replace_file_tags  # type: ignore[attr-defined]
            module.update_fts = update_fts  # type: ignore[attr-defined]
            module.upsert_file = upsert_file  # type: ignore[attr-defined]
            module.upsert_tags = upsert_tags_repo  # type: ignore[attr-defined]
            module.get_file_by_path = get_file_by_path  # type: ignore[attr-defined]
            module.list_untagged_under_path = list_untagged_under_path  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        setattr(sys.modules["db"], name, module)  # type: ignore[attr-defined]

    if "core.pipeline.testhooks" not in sys.modules:
        testhooks_module = types.ModuleType("core.pipeline.testhooks")

        @dataclass
        class TaggingDeps:  # pragma: no cover - minimal stub
            dbwriter_factory: object | None = None
            loader_factory: object | None = None
            quiesce: object | None = None
            conn_factory: Callable[[str], object] | None = None

        testhooks_module.TaggingDeps = TaggingDeps
        sys.modules["core.pipeline.testhooks"] = testhooks_module


_install_stub_modules()

from core.pipeline.contracts import DBItem
from core.pipeline.stages import write_stage as write_stage_module
from core.pipeline.stages.tag_stage import TagStageResult
from core.pipeline.stages.write_stage import WriteStage
from core.pipeline.types import IndexPhase, PipelineContext, ProgressEmitter


@dataclass
class _FakeWriter:
    """Minimal stand-in for the real database writer."""

    progress_cb: Callable[[str, int, int], None]
    total_count: int
    fail_after: int | None = None
    start_called: bool = False
    stop_calls: List[tuple[bool, bool]] = None
    put_count: int = 0

    def __post_init__(self) -> None:
        self.stop_calls = []

    def start(self) -> None:
        self.start_called = True

    def raise_if_failed(self) -> None:  # pragma: no cover - behaviourally inert
        return None

    def put(self, item) -> None:  # pragma: no cover - behaviourally inert
        next_count = self.put_count + 1
        if self.fail_after is not None and next_count > self.fail_after:
            raise RuntimeError("simulated put failure")
        self.put_count = next_count
        self.progress_cb("fts.progress", self.put_count, self.total_count)

    def stop(self, *, flush: bool = True, wait_forever: bool = False) -> None:
        self.stop_calls.append((flush, wait_forever))


class _FakeDeps:
    """Fake dependency provider capturing quiesce handling."""

    def __init__(
        self,
        *,
        total_count: int,
        build_writer_error: Exception | None = None,
        writer_fail_after: int | None = None,
    ) -> None:
        self.total_count = total_count
        self.build_writer_error = build_writer_error
        self.writer_fail_after = writer_fail_after
        self.begin_calls = 0
        self.end_calls = 0
        self.connect_paths: list[str] = []
        self.writer: _FakeWriter | None = None

    def build_writer(self, *, ctx: PipelineContext, progress_cb):
        if self.build_writer_error is not None:
            raise self.build_writer_error
        self.writer = _FakeWriter(progress_cb, self.total_count, self.writer_fail_after)
        return self.writer

    def begin_quiesce(self) -> None:
        self.begin_calls += 1

    def end_quiesce(self) -> None:
        self.end_calls += 1

    def connect(self, db_path: str) -> None:
        self.connect_paths.append(db_path)


def _make_context(db_path: Path, *, is_cancelled) -> PipelineContext:
    return PipelineContext(
        db_path=db_path,
        settings=SimpleNamespace(),
        thresholds={},
        max_tags_map={},
        tagger_sig="sig",
        tagger_override=None,
        progress_cb=None,
        is_cancelled=is_cancelled,
    )


def _make_tag_result(count: int) -> TagStageResult:
    items = [
        DBItem(file_id=i, tags=(), width=None, height=None, tagger_sig=None, tagged_at=None)
        for i in range(count)
    ]
    return TagStageResult(records=[], db_items=items, tagged_count=count)


@pytest.fixture(autouse=True)
def _patch_sleep(monkeypatch):
    monkeypatch.setattr(write_stage_module.time, "sleep", lambda *_: None)


def test_write_stage_handles_writer_failure(monkeypatch, tmp_path):
    settle_calls: list[str] = []
    monkeypatch.setattr(
        write_stage_module,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(path),
    )

    total = 4
    deps = _FakeDeps(total_count=total, writer_fail_after=2)
    stage = WriteStage(deps=deps)

    progress_events: list[IndexProgress] = []
    emitter = ProgressEmitter(progress_events.append)

    ctx = _make_context(tmp_path / "db.sqlite", is_cancelled=lambda: False)
    tag_result = _make_tag_result(total)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 2
    assert result.fts_processed == 2
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert settle_calls == [str(ctx.db_path)]
    assert deps.writer is not None
    assert deps.writer.stop_calls == [(True, True)]


def test_write_stage_handles_build_writer_failure(monkeypatch, tmp_path):
    settle_calls: list[str] = []
    monkeypatch.setattr(
        write_stage_module,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(path),
    )

    deps = _FakeDeps(total_count=3, build_writer_error=RuntimeError("boom"))
    stage = WriteStage(deps=deps)

    emitter = ProgressEmitter(lambda _: None)
    ctx = _make_context(tmp_path / "db.sqlite", is_cancelled=lambda: False)
    tag_result = _make_tag_result(3)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 0
    assert result.fts_processed == 0
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert settle_calls == [str(ctx.db_path)]


def test_write_stage_respects_cancellation(monkeypatch, tmp_path):
    settle_calls: list[str] = []
    monkeypatch.setattr(
        write_stage_module,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(path),
    )

    total = 3
    deps = _FakeDeps(total_count=total)
    stage = WriteStage(deps=deps)

    progress_events: list[IndexProgress] = []
    emitter = ProgressEmitter(progress_events.append)

    cancel_calls = 0

    def _cancel() -> bool:
        nonlocal cancel_calls
        cancel_calls += 1
        return cancel_calls >= 2

    ctx = _make_context(tmp_path / "db.sqlite", is_cancelled=_cancel)
    tag_result = _make_tag_result(total)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 1
    assert result.fts_processed == 1
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert settle_calls == [str(ctx.db_path), str(ctx.db_path)]
    assert deps.writer is not None
    assert deps.writer.stop_calls == [(True, True)]
    assert progress_events
    assert progress_events[-1].phase == IndexPhase.FTS
    assert progress_events[-1].done == result.written
    assert progress_events[-1].total == total
