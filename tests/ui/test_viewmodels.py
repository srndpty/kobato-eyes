"""Unit tests for UI view models executable in headless mode."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import pytest
from PyQt6.QtCore import QCoreApplication

import ui.viewmodels.main_view_model as main_view_model_module
from core.config import PipelineSettings, TaggerSettings
from core.pipeline import RetagResult
from dup.scanner import DuplicateScanConfig
from ui.viewmodels import DupViewModel, MainViewModel, SettingsViewModel, TagsSearchState, TagsViewModel


@pytest.fixture(scope="session", autouse=True)
def _headless_qapp() -> Iterable[None]:
    os.environ.setdefault("KOE_HEADLESS", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield


def test_tags_view_model_wraps_dependencies(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_run_index_once(db_path: Path, **kwargs):
        calls["run"] = (db_path, kwargs)
        return {"files": 3}

    def fake_scan_and_tag(folder: Path, **kwargs):
        calls["scan"] = (folder, kwargs)
        return {"scanned": 5}

    view_model = TagsViewModel(
        db_path=tmp_path / "app.db",
        connection_factory=lambda path: f"conn:{Path(path)}",
        run_index_once=fake_run_index_once,
        scan_and_tag=fake_scan_and_tag,
        retag_query=lambda db, predicate, params: RetagResult(len(params), [1, 2][: len(params)]),
        retag_all=lambda db, force, settings: RetagResult(42, [3, 4]),
        run_retag_selection=lambda *args, **kwargs: {"tagged": 99},
        load_settings=lambda: PipelineSettings(),
        load_thresholds=lambda conn: {"general": 0.5},
        list_tag_names=lambda conn: ["foo"],
        mark_files_absent=lambda conn, ids: calls.setdefault("marked_absent", (conn, list(ids))),
        search_files=lambda conn, where, params, **kwargs: [{"id": 1}],
        iter_paths_for_search=lambda conn, query: ["/a", "/b"],
        ensure_directories=lambda: calls.setdefault("ensured", True),
        make_export_dir=lambda query: Path("/exports") / query,
        translate_query=lambda query, **kwargs: {"where": query, "kwargs": kwargs},
        extract_positive_terms=lambda query: {query},
    )

    result = view_model.run_index_once(view_model.db_path, settings=None)
    assert result == {"files": 3}
    assert calls["run"][0] == view_model.db_path

    scan_result = view_model.scan_and_tag(Path("/tmp"), recursive=False)
    assert scan_result["scanned"] == 5
    assert calls["scan"][0] == Path("/tmp")

    export_dir = view_model.make_export_dir("query")
    assert export_dir.name == "query"

    conn = view_model.open_connection()
    assert conn == f"conn:{view_model.db_path}"

    rows = view_model.search_files(conn, "1=1", [])
    assert rows == [{"id": 1}]
    paths = view_model.iter_paths_for_search(conn, "*")
    assert paths == ["/a", "/b"]
    view_model.mark_files_absent(conn, [10, 11])
    assert calls["marked_absent"] == (conn, [10, 11])

    retagged = view_model.retag_query(view_model.db_path, "predicate", [1, 2])
    assert isinstance(retagged, RetagResult)
    assert retagged.affected == 2
    all_retagged = view_model.retag_all(view_model.db_path, force=True, settings=PipelineSettings())
    assert all_retagged.affected == 42
    stats = view_model.run_retag_selection(view_model.db_path, [1, 2], settings=PipelineSettings())
    assert stats == {"tagged": 99}

    fragment = view_model.translate_query("rating:safe", file_alias="f")
    assert fragment["where"] == "rating:safe"
    assert "file_alias" in fragment["kwargs"]

    positives = view_model.extract_positive_terms("foo")
    assert positives == {"foo"}

    view_model.ensure_directories()
    assert calls["ensured"] is True


def test_tags_search_state_tracks_paging_and_generations() -> None:
    state = TagsSearchState(offset=12, busy=True, can_load_more=True, last_cancelled=True)

    state.begin_query()

    assert state.offset == 0
    assert state.reset_pending is True
    assert state.received_any is False
    assert state.last_cancelled is False
    assert state.can_load_more is False

    first_generation = state.begin_worker(reset=True)
    second_generation = state.begin_worker(reset=False)

    assert first_generation == 1
    assert second_generation == 2
    assert state.generations_reset == {1: True, 2: False}

    state.consume_rows(3, chunk_size=3)
    assert state.offset == 3
    assert state.received_any is True
    assert state.can_load_more is True

    state.consume_rows(1, chunk_size=3)
    assert state.offset == 4
    assert state.can_load_more is False

    assert state.finish_generation(first_generation) is True
    state.discard_generation(second_generation)
    assert state.generations_reset == {}


def test_tags_view_model_retag_all_keyword_arguments(tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def fake_retag_all(db_path: Path, *, force: bool, settings: PipelineSettings) -> RetagResult:
        recorded["db_path"] = db_path
        recorded["force"] = force
        recorded["settings"] = settings
        return RetagResult(7, [1, 2, 3])

    view_model = TagsViewModel(
        db_path=tmp_path / "tags.db",
        connection_factory=lambda path: f"conn:{Path(path)}",
        run_index_once=lambda *args, **kwargs: {},
        scan_and_tag=lambda *args, **kwargs: {},
        retag_query=lambda *args, **kwargs: RetagResult(0, []),
        run_retag_selection=lambda *args, **kwargs: {},
        retag_all=fake_retag_all,
        load_settings=lambda: PipelineSettings(),
        load_thresholds=lambda conn: {},
        list_tag_names=lambda conn: [],
        search_files=lambda *args, **kwargs: [],
        iter_paths_for_search=lambda *args, **kwargs: [],
        ensure_directories=lambda: None,
        make_export_dir=lambda query: tmp_path,
        translate_query=lambda query, **kwargs: {},
        extract_positive_terms=lambda query: set(),
    )

    settings = PipelineSettings()
    result = view_model.retag_all(view_model.db_path, force=True, settings=settings)

    assert result.affected == 7
    assert recorded["db_path"] == view_model.db_path
    assert recorded["force"] is True
    assert recorded["settings"] is settings


def test_tags_view_model_refresh_roots_aggregates_and_stops_on_cancel(tmp_path: Path) -> None:
    calls: list[Path] = []

    def fake_scan_and_tag(folder: Path, **kwargs):
        calls.append(folder)
        return {
            "queued": 2,
            "tagged": 1,
            "elapsed_sec": 0.5,
            "missing": 1,
            "soft_deleted": 1,
            "hard_deleted": 0,
            "cancelled": folder.name == "second",
        }

    view_model = TagsViewModel(
        db_path=tmp_path / "tags.db",
        connection_factory=lambda path: None,
        run_index_once=lambda *args, **kwargs: {},
        scan_and_tag=fake_scan_and_tag,
        retag_query=lambda *args, **kwargs: RetagResult(0, []),
        run_retag_selection=lambda *args, **kwargs: {},
        retag_all=lambda *args, **kwargs: RetagResult(0, []),
        load_settings=lambda: PipelineSettings(),
        load_thresholds=lambda conn: {},
        list_tag_names=lambda conn: [],
        search_files=lambda *args, **kwargs: [],
        iter_paths_for_search=lambda *args, **kwargs: [],
        ensure_directories=lambda: None,
        make_export_dir=lambda query: tmp_path,
        translate_query=lambda query, **kwargs: {},
        extract_positive_terms=lambda query: set(),
    )

    result = view_model.refresh_roots(
        [Path("first"), Path("second"), Path("third")],
        settings=PipelineSettings(batch_size=4),
        hard_delete_missing=True,
    )

    assert calls == [Path("first"), Path("second")]
    assert result["queued"] == 4.0
    assert result["tagged"] == 2.0
    assert result["missing"] == 2.0
    assert result["soft_deleted"] == 2.0
    assert result["hard_delete"] is True
    assert result["cancelled"] is True
    assert len(result["roots"]) == 2


def test_tags_view_model_refresh_roots_handles_cancel_callback_exception(tmp_path: Path) -> None:
    view_model = TagsViewModel(
        db_path=tmp_path / "tags.db",
        connection_factory=lambda path: None,
        run_index_once=lambda *args, **kwargs: {},
        scan_and_tag=lambda *args, **kwargs: {"queued": 1},
        retag_query=lambda *args, **kwargs: RetagResult(0, []),
        run_retag_selection=lambda *args, **kwargs: {},
        retag_all=lambda *args, **kwargs: RetagResult(0, []),
        load_settings=lambda: PipelineSettings(),
        load_thresholds=lambda conn: {},
        list_tag_names=lambda conn: [],
        search_files=lambda *args, **kwargs: [],
        iter_paths_for_search=lambda *args, **kwargs: [],
        ensure_directories=lambda: None,
        make_export_dir=lambda query: tmp_path,
        translate_query=lambda query, **kwargs: {},
        extract_positive_terms=lambda query: set(),
    )

    result = view_model.refresh_roots(
        [Path("never-scanned")],
        settings=PipelineSettings(),
        is_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    assert result["cancelled"] is True
    assert result["roots"] == []


def test_dup_view_model_cluster_and_thumbnails(tmp_path: Path) -> None:
    generated: dict[str, object] = {}

    class DummyScanner:
        def __init__(self, config: DuplicateScanConfig) -> None:
            self.config = config

        def build_clusters(self, files: Iterable[Sequence[object]]):
            generated["files"] = list(files)
            return ["cluster"]

    view_model = DupViewModel(
        db_path=tmp_path / "dup.db",
        connection_factory=lambda path: f"conn:{Path(path)}",
        iter_files_for_dup=lambda conn, path_like: [[1, "file"]],
        mark_files_absent=lambda conn, ids: generated.setdefault("marked", ids),
        scanner_factory=lambda config: DummyScanner(config),
        generate_thumbnail=lambda path, cache_dir, size, format: generated.setdefault(
            "thumb", (Path(path), Path(cache_dir), size, format)
        ),
        get_thumbnail=lambda path, w, h: f"thumb:{Path(path)}:{w}x{h}",
        cache_dir_factory=lambda: tmp_path / "cache",
    )

    conn = view_model.open_connection()
    assert conn == f"conn:{view_model.db_path}"

    clusters = view_model.build_clusters(DuplicateScanConfig(hamming_threshold=1, size_ratio=0.9), [[1, 2]])
    assert clusters == ["cluster"]
    assert generated["files"] == [[1, 2]]

    cache_dir = view_model.thumbnail_cache_dir()
    assert cache_dir == tmp_path / "cache" / "thumbs"

    view_model.generate_thumbnail(Path("img.jpg"), cache_dir, size=(128, 128))
    assert generated["thumb"] == (Path("img.jpg"), cache_dir, (128, 128), "WEBP")

    pix = view_model.get_thumbnail(Path("img.jpg"), 64, 64)
    assert pix == "thumb:img.jpg:64x64"

    view_model.mark_files_absent("conn", [1, 2, 3])
    assert generated["marked"] == [1, 2, 3]


def test_settings_view_model_build_and_reset(tmp_path: Path) -> None:
    emitted: list[PipelineSettings] = []
    reset_calls: list[tuple[Path, bool]] = []

    def fake_reset(path: Path, *, backup: bool) -> dict[str, object]:
        reset_calls.append((path, backup))
        return {"backup_paths": [str(path)]}

    view_model = SettingsViewModel(
        db_path=tmp_path / "settings.db",
        reset_database=fake_reset,
        provider_loader=lambda: ["CUDA", "CPU"],
    )
    view_model.settings_applied.connect(emitted.append)

    previous = TaggerSettings()
    settings = view_model.build_settings(
        roots=[Path("/data")],
        excluded=[Path("/ignore")],
        batch_size=16,
        prefetch_depth=6,
        tagger_name="wd14-onnx",
        model_path="/model.onnx",
        device="cpu",
        previous_tagger=previous,
    )
    view_model.apply_settings(settings)

    assert [Path(p) for p in emitted[-1].roots] == [Path("/data")]
    assert Path(emitted[-1].tagger.model_path).name == "model.onnx"
    assert emitted[-1].tagger.device == "cpu"
    assert emitted[-1].batch_size == 16
    assert emitted[-1].prefetch_depth == 6

    message = view_model.check_tagger_environment()
    assert "ONNX providers" in message

    result = view_model.reset_database(backup=True)
    assert result["backup_paths"] == [str(tmp_path / "settings.db")]
    assert reset_calls == [(tmp_path / "settings.db", True)]

    view_model.set_current_settings(settings)
    assert view_model.current_settings == settings


def test_settings_view_model_reset_failure(tmp_path: Path) -> None:
    view_model = SettingsViewModel(
        db_path=tmp_path / "err.db",
        reset_database=lambda path, *, backup: (_ for _ in ()).throw(RuntimeError("boom")),
        provider_loader=lambda: [],
    )

    errors: list[str] = []
    view_model.database_reset_failed.connect(errors.append)

    with pytest.raises(RuntimeError):
        view_model.reset_database(backup=False)

    assert errors and errors[0] == "boom"


def test_main_view_model_bootstrap_and_factories(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []
    settings = PipelineSettings(batch_size=12)
    db_path = tmp_path / "main.db"

    monkeypatch.setattr(main_view_model_module, "migrate_data_dir_if_needed", lambda: calls.append("migrate"))
    monkeypatch.setattr(main_view_model_module, "ensure_dirs", lambda: calls.append("ensure"))
    monkeypatch.setattr(main_view_model_module, "get_db_path", lambda: db_path)
    monkeypatch.setattr(main_view_model_module, "_quick_settle_sqlite", lambda path: calls.append(("settle", path)))
    monkeypatch.setattr(main_view_model_module, "bootstrap_if_needed", lambda path: calls.append(("bootstrap", path)))
    monkeypatch.setattr(main_view_model_module, "load_settings", lambda: settings)

    saved: list[PipelineSettings] = []
    monkeypatch.setattr(main_view_model_module, "save_settings", saved.append)

    view_model = MainViewModel()
    emitted: list[PipelineSettings] = []
    view_model.settings_changed.connect(emitted.append)

    assert view_model.db_path == db_path
    assert view_model.current_settings is settings
    assert calls == ["migrate", "ensure", ("settle", db_path), ("bootstrap", db_path)]
    assert isinstance(view_model.create_tags_view_model(), TagsViewModel)
    assert isinstance(view_model.create_dup_view_model(), DupViewModel)
    settings_view_model = view_model.create_settings_view_model()
    assert isinstance(settings_view_model, SettingsViewModel)

    new_settings = PipelineSettings(batch_size=3)
    view_model.apply_settings(new_settings)

    assert saved == [new_settings]
    assert emitted == [new_settings]
    assert view_model.current_settings is new_settings
