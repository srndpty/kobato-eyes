from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.modules.pop("core.pipeline", None)
sys.modules.pop("core", None)
sys.modules.pop("db.connection", None)
sys.modules.pop("db", None)
sys.modules.pop("db.schema", None)
sys.modules.pop("db.repository", None)

from core.config import AppPaths, PipelineSettings
from core.pipeline import scan_and_tag
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


def _make_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scan_and_tag_missing_root_returns_empty_stats(tmp_path: Path) -> None:
    missing_root = tmp_path / "does-not-exist"

    stats = scan_and_tag(missing_root)

    assert stats == {
        "queued": 0,
        "tagged": 0,
        "elapsed_sec": 0.0,
        "missing": 0,
        "soft_deleted": 0,
        "hard_deleted": 0,
    }


def test_scan_and_tag_unsupported_extension_returns_early(
    temp_env: Path, tmp_path: Path
) -> None:
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


def test_scan_and_tag_deduplicates_paths(
    temp_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    image_path = root / "untagged.png"
    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    image_path.write_bytes(png_bytes)

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

    call_args: list[Path] = []

    def fake_run_tag_job(*args, **kwargs):
        call_args.append(Path(args[1]))
        return object()

    monkeypatch.setattr("core.pipeline.manual_refresh.run_tag_job", fake_run_tag_job)
    monkeypatch.setattr("core.pipeline.manual_refresh._resolve_tagger", lambda *a, **k: object())

    stats = scan_and_tag(root)

    assert stats["queued"] == 1
    assert stats["tagged"] == 1
    assert len(call_args) == 1
    assert call_args[0] == image_path.resolve()


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
