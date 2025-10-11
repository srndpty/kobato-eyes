"""Tests for retagging helpers in the pipeline layer."""

from __future__ import annotations

import enum
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_CORE_DIR = SRC_ROOT / "core"
_PIPELINE_DIR = _CORE_DIR / "pipeline"


def _ensure_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
        submodule_search_locations=[str(path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_pipeline_modules() -> tuple[types.ModuleType, types.ModuleType]:
    if "core" not in sys.modules:
        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = [str(_CORE_DIR)]
        sys.modules["core"] = core_pkg

    if "core.config" not in sys.modules:
        config_mod = types.ModuleType("core.config")

        class _TaggerSettingsStub:
            def __init__(
                self,
                *,
                name: str = "dummy",
                thresholds: dict[str, float] | None = None,
                model_path: str | None = None,
                tags_csv: str | None = None,
                max_tags: dict[str, int] | None = None,
            ) -> None:
                self.name = name
                self.thresholds = thresholds or {"general": 0.35}
                self.model_path = model_path
                self.tags_csv = tags_csv
                self.max_tags = max_tags

        class _PipelineSettingsStub:
            def __init__(self, tagger: _TaggerSettingsStub | None = None) -> None:
                self.tagger = tagger or _TaggerSettingsStub()

        def _load_settings() -> _PipelineSettingsStub:
            return _PipelineSettingsStub()

        config_mod.TaggerSettings = _TaggerSettingsStub
        config_mod.PipelineSettings = _PipelineSettingsStub
        config_mod.load_settings = _load_settings
        sys.modules["core.config"] = config_mod

    if "tagger.base" not in sys.modules:
        tagger_base = types.ModuleType("tagger.base")

        class _TagCategory(enum.IntEnum):
            GENERAL = 0
            CHARACTER = 1
            RATING = 2
            COPYRIGHT = 3
            ARTIST = 4
            META = 5

        tagger_base.TagCategory = _TagCategory
        sys.modules["tagger.base"] = tagger_base

    if "core.pipeline" not in sys.modules:
        pipeline_pkg = types.ModuleType("core.pipeline")
        pipeline_pkg.__path__ = [str(_PIPELINE_DIR)]
        sys.modules["core.pipeline"] = pipeline_pkg

    signature_module = sys.modules.get("core.pipeline.signature")
    if signature_module is None:
        signature_module = _ensure_module("core.pipeline.signature", _PIPELINE_DIR / "signature.py")

    retag_module = sys.modules.get("core.pipeline.retag")
    if retag_module is None:
        retag_module = _ensure_module("core.pipeline.retag", _PIPELINE_DIR / "retag.py")

    return retag_module, signature_module


_RETAG_MODULE, _SIGNATURE_MODULE = _prepare_pipeline_modules()
retag_query = _RETAG_MODULE.retag_query
retag_all = _RETAG_MODULE.retag_all
current_tagger_sig = _SIGNATURE_MODULE.current_tagger_sig

from db.connection import get_conn  # noqa: E402
from db.schema import apply_schema  # noqa: E402


def _prepare_database(db_path: Path) -> None:
    conn = get_conn(db_path)
    try:
        apply_schema(conn)
    finally:
        conn.close()


def test_retag_query_resets_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"
    _prepare_database(db_path)

    conn = get_conn(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO files (path, tagger_sig, last_tagged_at) VALUES (?, ?, ?)",
            ("sample.png", "legacy-signature", 123.456),
        )
        file_id = cursor.lastrowid
        conn.commit()
    finally:
        conn.close()

    updated = retag_query(db_path, "id = ?", [file_id])

    assert updated == 1

    conn = get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT tagger_sig, last_tagged_at FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row["tagger_sig"] is None
    assert row["last_tagged_at"] is None


def test_retag_all_filters_by_signature_and_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "library.db"
    _prepare_database(db_path)

    config_module = sys.modules["core.config"]
    settings = config_module.PipelineSettings(config_module.TaggerSettings(name="dummy"))
    signature = current_tagger_sig(settings)

    conn = get_conn(db_path)
    try:
        row1 = conn.execute(
            "INSERT INTO files (path, tagger_sig, last_tagged_at) VALUES (?, ?, ?)",
            ("match.png", signature, 10.0),
        ).lastrowid
        row2 = conn.execute(
            "INSERT INTO files (path, tagger_sig, last_tagged_at) VALUES (?, ?, ?)",
            ("mismatch.png", "other-signature", 20.0),
        ).lastrowid
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(_RETAG_MODULE, "load_settings", lambda: settings)

    updated = retag_all(db_path, force=False)

    assert updated == 1

    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, tagger_sig, last_tagged_at FROM files ORDER BY id",
        ).fetchall()
    finally:
        conn.close()

    row_map = {row["id"]: row for row in rows}
    assert row_map[row1]["tagger_sig"] is None
    assert row_map[row1]["last_tagged_at"] is None
    assert row_map[row2]["tagger_sig"] == "other-signature"
    assert row_map[row2]["last_tagged_at"] == 20.0

    conn = get_conn(db_path)
    try:
        conn.execute(
            "UPDATE files SET tagger_sig = ?, last_tagged_at = ? WHERE id = ?",
            (signature, 99.0, row1),
        )
        conn.execute(
            "UPDATE files SET tagger_sig = ?, last_tagged_at = ? WHERE id = ?",
            ("other-signature", 77.0, row2),
        )
        conn.commit()
    finally:
        conn.close()

    updated_force = retag_all(db_path, force=True)

    assert updated_force == 2

    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, tagger_sig, last_tagged_at FROM files ORDER BY id",
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        assert row["tagger_sig"] is None
        assert row["last_tagged_at"] is None
