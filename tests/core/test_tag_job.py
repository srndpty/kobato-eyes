"""Tests for the tagging job pipeline."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pytest

def _ensure_core_config_stub() -> None:
    if "core.config" in sys.modules:
        return

    core_pkg = sys.modules.get("core")
    if core_pkg is None:
        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["core"] = core_pkg

    config_module = types.ModuleType("core.config")

    class AppPaths:
        def __init__(self, root: Path | None = None) -> None:
            self._root = root or (Path.cwd() / "_kobato_stub_data")

        def data_dir(self) -> Path:
            return self._root

        def db_path(self) -> Path:
            return self._root / "kobato-eyes.db"

        def index_dir(self) -> Path:
            return self._root / "index"

        def cache_dir(self) -> Path:
            return self._root / "cache"

        def log_dir(self) -> Path:
            return self._root / "log"

        def ensure_data_dirs(self) -> None:
            for path in [
                self.data_dir(),
                self.index_dir(),
                self.cache_dir(),
                self.log_dir(),
            ]:
                path.mkdir(parents=True, exist_ok=True)

        def migrate_data_dir_if_needed(self) -> bool:
            return False

    config_module.AppPaths = AppPaths
    config_module.__all__ = ["AppPaths"]

    sys.modules["core.config"] = config_module
    setattr(core_pkg, "config", config_module)


def _ensure_pillow_stub() -> None:
    if "PIL" in sys.modules:
        return

    pil_pkg = types.ModuleType("PIL")
    pil_pkg.__path__ = []  # type: ignore[attr-defined]

    class _StubImage:
        def __init__(self, path: str | None = None) -> None:
            self.path = path
            self.mode = "RGB"
            self.size = (0, 0)
            self.width = 0
            self.height = 0

        def draft(self, *_args, **_kwargs) -> None:  # pragma: no cover - safety stub
            return None

        def load(self) -> "_StubImage":  # pragma: no cover - safety stub
            return self

        def thumbnail(self, *_args, **_kwargs) -> None:  # pragma: no cover - safety stub
            return None

        def convert(self, _mode: str) -> "_StubImage":  # pragma: no cover - safety stub
            return self

        def close(self) -> None:  # pragma: no cover - safety stub
            return None

    image_module = types.ModuleType("PIL.Image")
    image_module._StubImage = _StubImage
    image_module.MAX_IMAGE_PIXELS = None
    image_module.LANCZOS = 1

    class _Resampling:
        LANCZOS = 1

    image_module.Resampling = _Resampling

    def _open(path: str) -> _StubImage:  # pragma: no cover - safety stub
        return _StubImage(path)

    image_module.open = _open
    image_module.new = lambda *_args, **_kwargs: _StubImage()

    class DecompressionBombError(Exception):
        pass

    image_module.DecompressionBombError = DecompressionBombError

    imagefile_module = types.ModuleType("PIL.ImageFile")
    imagefile_module.LOAD_TRUNCATED_IMAGES = False

    class UnidentifiedImageError(Exception):
        pass

    pil_pkg.Image = image_module
    pil_pkg.ImageFile = imagefile_module
    pil_pkg.UnidentifiedImageError = UnidentifiedImageError

    sys.modules["PIL"] = pil_pkg
    sys.modules["PIL.Image"] = image_module
    sys.modules["PIL.ImageFile"] = imagefile_module


def _ensure_tagger_stub() -> None:
    if "tagger.base" in sys.modules:
        return

    tagger_pkg = types.ModuleType("tagger")
    tagger_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["tagger"] = tagger_pkg

    tagger_base = types.ModuleType("tagger.base")

    class TagCategory(IntEnum):
        GENERAL = 0
        CHARACTER = 1
        RATING = 2
        COPYRIGHT = 3
        ARTIST = 4
        META = 5

    @dataclass(frozen=True)
    class TagPrediction:
        name: str
        score: float
        category: TagCategory

    @dataclass
    class TagResult:
        tags: list[TagPrediction]

    ThresholdMap = Mapping[TagCategory, float]
    MaxTagsMap = Mapping[TagCategory, int]

    class ITagger(Protocol):
        def infer_batch(
            self,
            images: Sequence[Any],
            *,
            thresholds: ThresholdMap | None = None,
            max_tags: MaxTagsMap | None = None,
        ) -> list[TagResult]:
            ...

    tagger_base.TagCategory = TagCategory
    tagger_base.TagPrediction = TagPrediction
    tagger_base.TagResult = TagResult
    tagger_base.ITagger = ITagger
    tagger_base.ThresholdMap = ThresholdMap
    tagger_base.MaxTagsMap = MaxTagsMap
    tagger_base.__all__ = [
        "ITagger",
        "TagCategory",
        "TagPrediction",
        "TagResult",
        "ThresholdMap",
        "MaxTagsMap",
    ]

    sys.modules["tagger.base"] = tagger_base
    tagger_pkg.base = tagger_base  # type: ignore[attr-defined]


def _load_tag_job_module():
    if "core.tag_job" in sys.modules:
        return sys.modules["core.tag_job"]

    root = Path(__file__).resolve().parents[2]
    module_path = root / "src" / "core" / "tag_job.py"

    if "core" not in sys.modules:
        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = [str(root / "src" / "core")]  # type: ignore[attr-defined]
        sys.modules["core"] = core_pkg

    spec = importlib.util.spec_from_file_location("core.tag_job", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load core.tag_job module for testing")

    module = importlib.util.module_from_spec(spec)
    sys.modules["core.tag_job"] = module
    spec.loader.exec_module(module)
    setattr(sys.modules["core"], "tag_job", module)
    return module


_ensure_core_config_stub()
_ensure_pillow_stub()
_ensure_tagger_stub()
tag_job_module = _load_tag_job_module()
TagJobConfig = tag_job_module.TagJobConfig
run_tag_job = tag_job_module.run_tag_job

tagger_base_module = sys.modules["tagger.base"]
ITagger = tagger_base_module.ITagger
TagCategory = tagger_base_module.TagCategory
TagPrediction = tagger_base_module.TagPrediction
TagResult = tagger_base_module.TagResult

from db.connection import get_conn
from db.schema import apply_schema


@dataclass
class _LoadedImage:
    width: int
    height: int


class DummyTagger(ITagger):
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict | None, dict | None]] = []

    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds=None,
        max_tags=None,
    ) -> list[TagResult]:
        self.calls.append((len(images), dict(thresholds or {}), dict(max_tags or {})))
        predictions = [
            TagPrediction(name="character:kobato", score=0.9, category=TagCategory.CHARACTER),
            TagPrediction(name="rating:safe", score=0.95, category=TagCategory.GENERAL),
        ]
        return [TagResult(tags=predictions)]


def _make_image(path: Path) -> None:
    path.write_bytes(b"dummy image data")


@pytest.fixture()
def memory_conn(tmp_path: Path) -> Iterable[sqlite3.Connection]:
    conn = get_conn(":memory:")
    apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_run_tag_job_persists_predictions(
    monkeypatch: pytest.MonkeyPatch,
    memory_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    source = tmp_path / "image.png"
    _make_image(source)

    monkeypatch.setattr("core.tag_job.safe_load_image", lambda _path: _LoadedImage(32, 32))

    tagger = DummyTagger()
    config = TagJobConfig(
        thresholds={TagCategory.GENERAL: 0.5},
        max_tags={TagCategory.GENERAL: 10},
    )

    output = run_tag_job(tagger, source, memory_conn, config=config)
    assert output is not None
    assert output.file_id > 0
    assert tagger.calls
    _, recorded_thresholds, recorded_max = tagger.calls[-1]
    assert recorded_thresholds == {TagCategory.GENERAL: 0.5}
    assert recorded_max == {TagCategory.GENERAL: 10}

    file_row = memory_conn.execute(
        "SELECT size, sha256 FROM files WHERE id = ?",
        (output.file_id,),
    ).fetchone()
    assert file_row is not None and file_row["sha256"]

    tag_rows = memory_conn.execute(
        "SELECT name, category FROM tags ORDER BY name",
    ).fetchall()
    assert {(row["name"], row["category"]) for row in tag_rows} == {
        ("character:kobato", TagCategory.CHARACTER),
        ("rating:safe", TagCategory.GENERAL),
    }

    tag_scores = memory_conn.execute(
        "SELECT score FROM file_tags ORDER BY score DESC",
    ).fetchall()
    assert len(tag_scores) == 2
    assert tag_scores[0]["score"] >= tag_scores[1]["score"]

    fts_row = memory_conn.execute(
        "SELECT rowid AS file_id FROM fts_files WHERE fts_files MATCH ?",
        ("kobato",),
    ).fetchone()
    assert fts_row is not None
    assert fts_row["file_id"] == output.file_id


def test_run_tag_job_returns_none_for_missing(memory_conn: sqlite3.Connection) -> None:
    tagger = DummyTagger()
    result = run_tag_job(tagger, Path("/nonexistent.png"), memory_conn)
    assert result is None
    assert not tagger.calls


def test_run_tag_job_returns_none_when_image_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    memory_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    source = tmp_path / "image.png"
    _make_image(source)

    monkeypatch.setattr("core.tag_job.safe_load_image", lambda _path: None)

    def _unexpected(*_args, **_kwargs):  # pragma: no cover - safety guard
        raise AssertionError("database functions should not be called")

    monkeypatch.setattr("core.tag_job.upsert_file", _unexpected)
    monkeypatch.setattr("core.tag_job.upsert_tags", _unexpected)
    monkeypatch.setattr("core.tag_job.replace_file_tags", _unexpected)
    monkeypatch.setattr("core.tag_job.update_fts", _unexpected)

    tagger = DummyTagger()
    result = run_tag_job(tagger, source, memory_conn)

    assert result is None
    assert not tagger.calls


def test_run_tag_job_returns_none_when_infer_batch_empty(
    monkeypatch: pytest.MonkeyPatch,
    memory_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    source = tmp_path / "image.png"
    _make_image(source)

    monkeypatch.setattr("core.tag_job.safe_load_image", lambda path: _LoadedImage(32, 32))

    def _unexpected(*_args, **_kwargs):  # pragma: no cover - safety guard
        raise AssertionError("database functions should not be called")

    monkeypatch.setattr("core.tag_job.upsert_file", _unexpected)
    monkeypatch.setattr("core.tag_job.upsert_tags", _unexpected)
    monkeypatch.setattr("core.tag_job.replace_file_tags", _unexpected)
    monkeypatch.setattr("core.tag_job.update_fts", _unexpected)

    tagger = DummyTagger()

    def _empty_infer(images, *, thresholds=None, max_tags=None):
        tagger.calls.append((len(images), dict(thresholds or {}), dict(max_tags or {})))
        return []

    monkeypatch.setattr(tagger, "infer_batch", _empty_infer)

    result = run_tag_job(tagger, source, memory_conn)

    assert result is None
    assert tagger.calls == [(1, {}, {})]


def test_run_tag_job_uses_config_values_for_upsert(
    monkeypatch: pytest.MonkeyPatch,
    memory_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    source = tmp_path / "image.png"
    _make_image(source)

    # Ensure we control the loaded image instance.
    monkeypatch.setattr("core.tag_job.safe_load_image", lambda path: _LoadedImage(32, 32))

    recorded_upsert_kwargs: dict[str, object] = {}

    def _fake_upsert_file(conn, **kwargs):
        nonlocal recorded_upsert_kwargs
        recorded_upsert_kwargs = kwargs
        return 99

    monkeypatch.setattr("core.tag_job.upsert_file", _fake_upsert_file)

    def _fake_upsert_tags(_conn, tag_defs):
        return {entry["name"]: index for index, entry in enumerate(tag_defs, start=1)}

    replace_calls: list[tuple[int, list[tuple[int, float]]]] = []
    update_calls: list[tuple[int, str | None]] = []

    def _fake_replace(conn, file_id, tag_scores):
        replace_calls.append((file_id, tag_scores))

    def _fake_update(conn, file_id, fts_text):
        update_calls.append((file_id, fts_text))

    monkeypatch.setattr("core.tag_job.upsert_tags", _fake_upsert_tags)
    monkeypatch.setattr("core.tag_job.replace_file_tags", _fake_replace)
    monkeypatch.setattr("core.tag_job.update_fts", _fake_update)

    tagged_at = 1234.5
    config = TagJobConfig(tagger_sig="dummy/1.0", tagged_at=tagged_at)

    monkeypatch.setattr("core.tag_job.time.time", lambda: 9876.5)

    tagger = DummyTagger()
    result = run_tag_job(tagger, source, memory_conn, config=config)

    assert result is not None
    assert recorded_upsert_kwargs["tagger_sig"] == "dummy/1.0"
    assert recorded_upsert_kwargs["last_tagged_at"] == tagged_at
    assert recorded_upsert_kwargs["path"] == str(source)
    assert replace_calls and replace_calls[0][0] == result.file_id
    assert update_calls and update_calls[0][0] == result.file_id
