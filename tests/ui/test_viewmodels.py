"""Unit tests for UI view models executable in headless mode."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import pytest
from PyQt6.QtCore import QCoreApplication

from core.config import PipelineSettings, TaggerSettings
from dup.scanner import DuplicateScanConfig
from ui.viewmodels import DupViewModel, SettingsViewModel, TagsViewModel


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
        retag_query=lambda db, predicate, params: len(params),
        retag_all=lambda db, force, settings: 42,
        load_settings=lambda: PipelineSettings(),
        load_thresholds=lambda conn: {"general": 0.5},
        list_tag_names=lambda conn: ["foo"],
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

    retagged = view_model.retag_query(view_model.db_path, "predicate", [1, 2])
    assert retagged == 2
    all_retagged = view_model.retag_all(view_model.db_path, force=True, settings=PipelineSettings())
    assert all_retagged == 42

    fragment = view_model.translate_query("rating:safe", file_alias="f")
    assert fragment["where"] == "rating:safe"
    assert "file_alias" in fragment["kwargs"]

    positives = view_model.extract_positive_terms("foo")
    assert positives == {"foo"}

    view_model.ensure_directories()
    assert calls["ensured"] is True


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
        provider_loader=lambda name: ["CUDA", "CPU"],
    )
    view_model.settings_applied.connect(emitted.append)

    previous = TaggerSettings()
    settings = view_model.build_settings(
        roots=[Path("/data")],
        excluded=[Path("/ignore")],
        batch_size=16,
        tagger_name="wd14-onnx",
        model_path="/model.onnx",
        previous_tagger=previous,
    )
    view_model.apply_settings(settings)

    assert [Path(p) for p in emitted[-1].roots] == [Path("/data")]
    assert Path(emitted[-1].tagger.model_path).name == "model.onnx"
    assert emitted[-1].batch_size == 16

    message = view_model.check_tagger_environment("wd14-onnx")
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
        provider_loader=lambda name: [],
    )

    errors: list[str] = []
    view_model.database_reset_failed.connect(errors.append)

    with pytest.raises(RuntimeError):
        view_model.reset_database(backup=False)

    assert errors and errors[0] == "boom"
