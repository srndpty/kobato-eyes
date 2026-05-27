"""Tests for manual scan-and-tag refresh behaviour."""

from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import core.pipeline.manual_refresh as manual_refresh
from core.config import AppPaths, PipelineSettings
from core.pipeline.manual_refresh import scan_and_tag
from db.connection import get_conn
from db.repository import replace_file_tags, upsert_file, upsert_tags
from utils import paths


@pytest.fixture()
def temp_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    app_paths = AppPaths(env={"KOE_DATA_DIR": str(data_dir)})
    monkeypatch.setattr(paths, "_APP_PATHS", app_paths)
    monkeypatch.setattr("core.pipeline.load_settings", lambda: PipelineSettings(allow_exts={".png"}))
    return data_dir


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, list] = {"tag": [], "write": []}

    class DummyTagStage:
        def run(self, ctx, emitter, records):  # noqa: ANN001 - signature defined by production class
            calls["tag"].append([Path(r.path).resolve() for r in records])
            db_items = [object() for _ in records]
            return SimpleNamespace(records=records, db_items=db_items, tagged_count=len(records))

    class DummyWriteStage:
        def run(self, ctx, emitter, tag_result):  # noqa: ANN001 - signature defined by production class
            calls["write"].append(list(tag_result.db_items))
            return SimpleNamespace(written=len(tag_result.db_items), fts_processed=len(tag_result.db_items))

    monkeypatch.setattr(manual_refresh, "TagStage", DummyTagStage)
    monkeypatch.setattr(manual_refresh, "WriteStage", DummyWriteStage)
    monkeypatch.setattr(manual_refresh, "_resolve_tagger", lambda *args, **kwargs: (object(), None, None))
    return calls


def _make_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_png(path: Path) -> None:
    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    path.write_bytes(png_bytes)


def test_scan_and_tag_missing_root_returns_empty_stats(tmp_path: Path) -> None:
    missing_root = tmp_path / "does-not-exist"

    stats = scan_and_tag(missing_root)

    # 実装では perf_counter を使うため elapsed_sec は 0.0 ちょうどではなく微小正値になり得る
    assert int(stats.get("queued", -1)) == 0
    assert int(stats.get("tagged", -1)) == 0
    assert int(stats.get("missing", -1)) == 0
    assert int(stats.get("soft_deleted", -1)) == 0
    assert int(stats.get("hard_deleted", -1)) == 0
    assert float(stats.get("elapsed_sec", -1.0)) >= 0.0


def test_scan_and_tag_unsupported_extension_returns_early(temp_env: Path, tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()
    unsupported = root / "note.txt"
    unsupported.write_text("hello", encoding="utf-8")

    stats = scan_and_tag(unsupported)

    assert stats["queued"] == 0
    assert stats["tagged"] == 0
    assert stats["missing"] == 0
    assert stats["soft_deleted"] == 0
    assert stats["hard_deleted"] == 0
    assert stats["elapsed_sec"] >= 0.0


def test_scan_and_tag_deduplicates_paths(temp_env: Path, tmp_path: Path, fake_pipeline: dict[str, list]) -> None:
    root = tmp_path / "library"
    root.mkdir()
    image_path = root / "untagged.png"
    _write_png(image_path)

    db_path = paths.get_db_path()
    conn = get_conn(db_path)
    try:
        stat_result = image_path.stat()
        upsert_file(
            conn,
            path=str(image_path.resolve()),
            size=stat_result.st_size,
            mtime=stat_result.st_mtime,
            sha256=_make_sha(image_path),
        )
    finally:
        conn.close()

    stats = scan_and_tag(root)

    assert stats["queued"] == 1
    assert stats["tagged"] == 1
    assert len(fake_pipeline["tag"]) == 1
    assert fake_pipeline["tag"][0] == [image_path.resolve()]


def test_scan_and_tag_propagates_tag_stage_failure(
    temp_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    image_path = root / "broken-tag.png"
    _write_png(image_path)

    class FailingTagStage:
        def run(self, ctx, emitter, records):  # noqa: ANN001 - signature defined by production class
            raise RuntimeError("tag stage exploded")

    monkeypatch.setattr(manual_refresh, "TagStage", FailingTagStage)

    with pytest.raises(RuntimeError, match="tag stage exploded"):
        scan_and_tag(root)


def test_scan_and_tag_treats_write_stage_failure_result_as_failure(
    temp_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    image_path = root / "broken-write.png"
    _write_png(image_path)

    class FailingWriteStage:
        def run(self, ctx, emitter, tag_result):  # noqa: ANN001 - signature defined by production class
            return SimpleNamespace(written=0, fts_processed=0, success=False, error="writer stopped")

    monkeypatch.setattr(manual_refresh, "WriteStage", FailingWriteStage)

    with pytest.raises(RuntimeError, match="writer stopped"):
        scan_and_tag(root)


def test_scan_and_tag_treats_write_stage_cancel_as_cancelled_stats(
    temp_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    image_path = root / "cancelled-write.png"
    _write_png(image_path)

    class CancellingWriteStage:
        def run(self, ctx, emitter, tag_result):  # noqa: ANN001 - signature defined by production class
            return SimpleNamespace(written=0, fts_processed=0, success=False, cancelled=True, error=None)

    monkeypatch.setattr(manual_refresh, "WriteStage", CancellingWriteStage)

    stats = scan_and_tag(root, is_cancelled=lambda: False)

    assert stats["cancelled"] is True
    assert stats["tagged"] == 0


def test_scan_and_tag_progress_callback_failure_is_best_effort(
    temp_env: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    image_path = root / "progress.png"
    _write_png(image_path)
    calls = 0

    def progress_cb(progress: manual_refresh.IndexProgress) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"progress failed at {progress.phase.name}")

    with caplog.at_level("ERROR", logger="core.pipeline.manual_refresh"):
        stats = scan_and_tag(root, progress_cb=progress_cb)

    assert stats["queued"] == 1
    assert stats["tagged"] == 1
    assert calls == 1
    assert "Refresh progress callback failed; disabling further updates" in caplog.text


def test_scan_and_tag_soft_deletes_missing(temp_env: Path, tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()
    present = root / "present.png"
    present.write_bytes(b"test")
    missing = root / "missing.png"

    db_path = paths.get_db_path()
    conn = get_conn(db_path)
    try:
        present_id = upsert_file(
            conn,
            path=str(present.resolve()),
            size=present.stat().st_size,
            mtime=present.stat().st_mtime,
            sha256=_make_sha(present),
        )
        tag_map = upsert_tags(conn, [{"name": "tag", "category": 0}])
        replace_file_tags(conn, present_id, [(tag_map["tag"], 0.9)])

        upsert_file(
            conn,
            path=str(missing.resolve()),
            size=123,
            mtime=456.0,
            sha256="deadbeef",
        )
    finally:
        conn.close()

    stats = scan_and_tag(root, hard_delete_missing=False)

    assert int(stats.get("missing", 0)) == 1
    assert int(stats.get("soft_deleted", 0)) == 1
    assert int(stats.get("hard_deleted", 0)) == 0

    conn2 = get_conn(db_path)
    try:
        present_row = conn2.execute(
            "SELECT is_present, deleted_at FROM files WHERE path = ?",
            (str(present.resolve()),),
        ).fetchone()
        assert present_row is not None
        assert int(present_row["is_present"]) == 1
        assert present_row["deleted_at"] is None

        missing_row = conn2.execute(
            "SELECT is_present, deleted_at FROM files WHERE path = ?",
            (str(missing.resolve()),),
        ).fetchone()
        assert missing_row is not None
        assert int(missing_row["is_present"]) == 0
        assert missing_row["deleted_at"] is not None
    finally:
        conn2.close()


def test_scan_and_tag_hard_deletes_missing(temp_env: Path, tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()
    missing = root / "gone.png"

    db_path = paths.get_db_path()
    conn = get_conn(db_path)
    try:
        file_id = upsert_file(
            conn,
            path=str(missing.resolve()),
            size=321,
            mtime=789.0,
            sha256="cafebabe",
        )
        gone_tag = upsert_tags(conn, [{"name": "gone", "category": 0}])["gone"]
        replace_file_tags(conn, file_id, [(gone_tag, 0.5)])
        conn.execute(
            "INSERT INTO signatures (file_id, phash_u64, dhash_u64) VALUES (?, 1, 2)",
            (file_id,),
        )
        conn.execute(
            "INSERT INTO fts_files (rowid, text) VALUES (?, ?)",
            (file_id, "gone"),
        )
    finally:
        conn.close()

    stats = scan_and_tag(root, hard_delete_missing=True)

    assert int(stats.get("missing", 0)) == 1
    assert int(stats.get("hard_deleted", 0)) == 1

    conn2 = get_conn(db_path)
    try:
        assert conn2.execute("SELECT 1 FROM files WHERE path = ?", (str(missing.resolve()),)).fetchone() is None
        assert conn2.execute("SELECT 1 FROM file_tags WHERE file_id = ?", (file_id,)).fetchone() is None
        assert conn2.execute("SELECT 1 FROM signatures WHERE file_id = ?", (file_id,)).fetchone() is None
        assert conn2.execute("SELECT 1 FROM fts_files WHERE rowid = ?", (file_id,)).fetchone() is None
    finally:
        conn2.close()


@pytest.mark.parametrize("hard_delete_missing", [False, True])
def test_scan_and_tag_cancelled_missing_cleanup_reports_processed_count(
    temp_env: Path,
    tmp_path: Path,
    hard_delete_missing: bool,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    db_path = paths.get_db_path()
    total_missing = 901

    conn = get_conn(db_path)
    try:
        for index in range(total_missing):
            upsert_file(
                conn,
                path=str((root / f"missing-{index:04d}.png").resolve()),
                size=100 + index,
                mtime=200.0 + index,
                sha256=f"deadbeef-{index}",
            )
    finally:
        conn.close()

    cancel_after_first_chunk = False

    def progress_cb(progress: manual_refresh.IndexProgress) -> None:
        nonlocal cancel_after_first_chunk
        if progress.phase is manual_refresh.IndexPhase.FTS and progress.done >= 900:
            cancel_after_first_chunk = True

    stats = scan_and_tag(
        root,
        hard_delete_missing=hard_delete_missing,
        progress_cb=progress_cb,
        is_cancelled=lambda: cancel_after_first_chunk,
    )

    assert stats["queued"] == 0
    assert stats["tagged"] == 0
    assert stats["missing"] == total_missing
    assert stats["cancelled"] is True
    if hard_delete_missing:
        assert stats["hard_deleted"] == 900
        assert stats["soft_deleted"] == 0
    else:
        assert stats["soft_deleted"] == 900
        assert stats["hard_deleted"] == 0

    conn2 = get_conn(db_path)
    try:
        remaining = conn2.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        absent = conn2.execute("SELECT COUNT(*) FROM files WHERE is_present = 0").fetchone()[0]
    finally:
        conn2.close()

    if hard_delete_missing:
        assert remaining == 1
        assert absent == 0
    else:
        assert remaining == total_missing
        assert absent == 900
