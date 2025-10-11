"""ProcessingPipeline enqueue scheduling unit tests."""

from __future__ import annotations

import enum
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest


class FakeSignal:
    """Minimal stand-in for Qt signal objects used in tests."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []

    def connect(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in list(self._callbacks):
            callback()


class FakeSignals:
    def __init__(self) -> None:
        self.finished = FakeSignal()


class FakeJobManager:
    def __init__(self) -> None:
        self.submissions: list[list[Path]] = []
        self.signals: list[FakeSignals] = []

    def submit(self, job, priority=None):  # type: ignore[no-untyped-def]
        self.submissions.append(list(job.items))
        signals = FakeSignals()
        self.signals.append(signals)
        return signals

    def shutdown(self) -> None:
        pass


@pytest.fixture()
def processing_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, FakeJobManager]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    monkeypatch.setenv("KOE_HEADLESS", "1")
    core_module = ModuleType("core")
    core_module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core", core_module)

    core_pipeline_module = ModuleType("core.pipeline")
    core_pipeline_module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core.pipeline", core_pipeline_module)

    dummy_config = ModuleType("core.config")

    class DummyAppPaths:
        def data_dir(self) -> Path:
            return Path.cwd()

        def index_dir(self) -> Path:
            return Path.cwd()

        def cache_dir(self) -> Path:
            return Path.cwd()

        def log_dir(self) -> Path:
            return Path.cwd()

        def db_path(self) -> Path:
            return Path.cwd() / "kobato-eyes.db"

        def ensure_data_dirs(self) -> None:
            pass

        def migrate_data_dir_if_needed(self) -> bool:
            return False

    class DummySettingsService:
        def __init__(self, _app_paths: DummyAppPaths | None = None) -> None:
            self.config_path = Path.cwd() / "config.yaml"

        def load(self) -> "DummyPipelineSettings":
            return DummyPipelineSettings()

        def save(self, _settings: "DummyPipelineSettings") -> None:
            pass

    class DummyPipelineSettings:
        def __init__(self, allow_exts: list[str] | None = None) -> None:
            self.allow_exts = allow_exts
            self.tagger = SimpleNamespace(thresholds={}, max_tags=None)

    dummy_config.PipelineSettings = DummyPipelineSettings
    dummy_config.AppPaths = DummyAppPaths
    dummy_config.SettingsService = DummySettingsService

    def _load_settings(*_args: Any, **_kwargs: Any) -> DummyPipelineSettings:
        return DummyPipelineSettings()

    dummy_config.load_settings = _load_settings
    monkeypatch.setitem(sys.modules, "core.config", dummy_config)
    dummy_tagger = ModuleType("tagger.base")

    class DummyTagCategory(enum.Enum):
        GENERAL = 0
        CHARACTER = 1
        RATING = 2
        COPYRIGHT = 3
        ARTIST = 4
        META = 5

    class DummyITagger:  # pragma: no cover - protocol stand-in
        pass

    dummy_tagger.TagCategory = DummyTagCategory
    dummy_tagger.ITagger = DummyITagger
    monkeypatch.setitem(sys.modules, "tagger.base", dummy_tagger)

    dummy_tag_job = ModuleType("core.tag_job")

    class DummyTagJobConfig:
        def __init__(
            self,
            thresholds: dict[Any, Any] | None = None,
            max_tags: dict[Any, Any] | None = None,
            tagger_sig: str | None = None,
        ) -> None:
            self.thresholds = thresholds
            self.max_tags = max_tags
            self.tagger_sig = tagger_sig

    def _run_tag_job(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Tag job should not execute in this test")

    dummy_tag_job.TagJobConfig = DummyTagJobConfig
    dummy_tag_job.run_tag_job = _run_tag_job
    monkeypatch.setitem(sys.modules, "core.tag_job", dummy_tag_job)

    dummy_pil = ModuleType("PIL")
    dummy_pil_image = ModuleType("PIL.Image")

    class _Image:
        pass

    dummy_pil_image.Image = _Image
    dummy_pil.Image = dummy_pil_image
    monkeypatch.setitem(sys.modules, "PIL", dummy_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", dummy_pil_image)

    dummy_signature = ModuleType("core.pipeline.signature")

    def _build_threshold_map(thresholds: dict[str, float] | None) -> dict[str, float]:
        return dict(thresholds or {})

    def _build_max_tags_map(max_tags: dict[str, int] | None) -> dict[str, int]:
        return dict(max_tags or {})

    def current_tagger_sig(*_args: Any, **_kwargs: Any) -> str:
        return "dummy-sig"

    dummy_signature._build_threshold_map = _build_threshold_map
    dummy_signature._build_max_tags_map = _build_max_tags_map
    dummy_signature.current_tagger_sig = current_tagger_sig
    monkeypatch.setitem(sys.modules, "core.pipeline.signature", dummy_signature)

    dummy_utils = ModuleType("utils.env")

    def _is_headless() -> bool:
        return True

    dummy_utils.is_headless = _is_headless
    monkeypatch.setitem(sys.modules, "utils.env", dummy_utils)

    dummy_paths = ModuleType("utils.paths")
    dummy_paths.ensure_dirs = lambda: None
    monkeypatch.setitem(sys.modules, "utils.paths", dummy_paths)

    dummy_db_connection = ModuleType("db.connection")

    class DummyConnection:
        def close(self) -> None:
            pass

    def _bootstrap_if_needed(_path: Path) -> None:
        return None

    def _get_conn(_path: Path) -> DummyConnection:
        return DummyConnection()

    dummy_db_connection.bootstrap_if_needed = _bootstrap_if_needed
    dummy_db_connection.get_conn = _get_conn
    monkeypatch.setitem(sys.modules, "db.connection", dummy_db_connection)

    dummy_jobs = ModuleType("core.jobs")

    class DummyBatchJob:
        def __init__(self, items: list[Path] | None = None) -> None:
            self.items = list(items or [])

    class DummyJobManager:
        def shutdown(self) -> None:
            pass

    dummy_jobs.BatchJob = DummyBatchJob
    dummy_jobs.JobManager = DummyJobManager
    monkeypatch.setitem(sys.modules, "core.jobs", dummy_jobs)

    dummy_scanner = ModuleType("core.scanner")
    dummy_scanner.DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    monkeypatch.setitem(sys.modules, "core.scanner", dummy_scanner)

    watcher_path = Path(__file__).resolve().parents[2] / "src" / "core" / "pipeline" / "watcher.py"
    spec = importlib.util.spec_from_file_location("core.pipeline.watcher", watcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load watcher module")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "core.pipeline"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "bootstrap_if_needed", lambda _path: None)
    monkeypatch.setattr(module, "ensure_dirs", lambda: None)
    fake_manager = FakeJobManager()
    pipeline = module.ProcessingPipeline(
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        job_manager=fake_manager,
        settings=None,
    )
    return pipeline, fake_manager


def test_enqueue_index_filters_duplicates_and_clears(
    processing_pipeline: tuple[Any, FakeJobManager], tmp_path: Path
) -> None:
    pipeline, manager = processing_pipeline
    jpg_path = tmp_path / "photo.JPG"
    other_path = tmp_path / "notes.txt"
    dup_path = tmp_path / "again.png"

    pipeline.enqueue_index([jpg_path])
    assert len(manager.submissions) == 1
    first_submission = manager.submissions[0]
    assert first_submission == [jpg_path.expanduser().resolve()]
    assert pipeline._scheduled == {jpg_path.expanduser().resolve()}

    manager.signals[0].finished.emit()
    assert pipeline._scheduled == set()

    pipeline.enqueue_index([other_path])
    assert len(manager.submissions) == 1
    assert pipeline._scheduled == set()

    pipeline.enqueue_index([dup_path, dup_path])
    assert len(manager.submissions) == 2
    second_submission = manager.submissions[1]
    assert second_submission == [dup_path.expanduser().resolve()]
    assert pipeline._scheduled == {dup_path.expanduser().resolve()}

    manager.signals[1].finished.emit()
    assert pipeline._scheduled == set()

    pipeline.enqueue_index([dup_path])
    assert pipeline._scheduled
    pipeline.stop()
    assert pipeline._scheduled == set()
