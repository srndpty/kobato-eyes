from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

try:  # pragma: no cover - helper stub for minimal test environment
    import pydantic as _pydantic  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - executed in CI without deps
    pydantic_stub = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **data: object) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: object) -> "_BaseModel":
            if isinstance(data, dict):
                return cls(**data)
            raise TypeError("Expected mapping for model validation")

        def model_dump(self) -> dict[str, object]:
            return dict(self.__dict__)

    def _identity_decorator(*args: object, **kwargs: object):
        def _decorator(func):
            return func

        return _decorator

    def _field(*, default: object = None, default_factory: object | None = None, **_: object) -> object:
        if default_factory is not None:
            try:
                return default_factory()
            except TypeError:
                return default_factory
        return default

    pydantic_stub.BaseModel = _BaseModel
    pydantic_stub.ConfigDict = dict
    pydantic_stub.Field = _field
    pydantic_stub.field_validator = _identity_decorator
    pydantic_stub.model_validator = _identity_decorator
    sys.modules["pydantic"] = pydantic_stub

if "yaml" not in sys.modules:  # pragma: no cover - helper stub when PyYAML is absent
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(_: object) -> dict[str, object]:
        return {}

    def _safe_dump(_: object, stream: object | None = None, **__: object) -> str:
        if hasattr(stream, "write"):
            stream.write("")
        return ""

    yaml_stub.safe_load = _safe_load
    yaml_stub.safe_dump = _safe_dump
    sys.modules["yaml"] = yaml_stub

if "numpy" not in sys.modules:  # pragma: no cover - helper stub for numpy-free CI
    numpy_stub = types.ModuleType("numpy")

    class _NDArray:  # pragma: no cover - simple placeholder
        pass

    def _array(values: object, dtype: object | None = None) -> object:  # noqa: ARG001 - signature parity
        return values

    numpy_stub.ndarray = _NDArray
    numpy_stub.array = _array
    numpy_stub.float32 = "float32"
    sys.modules["numpy"] = numpy_stub

if "PIL" not in sys.modules:  # pragma: no cover - helper stub for Pillow-free CI
    pil_stub = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")
    pil_image_file_module = types.ModuleType("PIL.ImageFile")

    class _DummyImage:  # pragma: no cover - placeholder for PIL.Image.Image
        pass

    class _DummyImageFile:  # pragma: no cover - placeholder for PIL.ImageFile
        pass

    pil_image_module.Image = _DummyImage
    pil_image_module.DecompressionBombError = type("DecompressionBombError", (RuntimeError,), {})
    pil_image_file_module.ImageFile = _DummyImageFile
    pil_stub.Image = pil_image_module
    pil_stub.ImageFile = pil_image_file_module
    pil_stub.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = pil_image_module
    sys.modules["PIL.ImageFile"] = pil_image_file_module

if "cv2" not in sys.modules:  # pragma: no cover - helper stub for opencv-free CI
    cv2_stub = types.ModuleType("cv2")

    def _identity(image: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
        return image

    cv2_stub.cvtColor = _identity
    cv2_stub.COLOR_BGR2RGB = 0
    sys.modules["cv2"] = cv2_stub

if "PyQt6" not in sys.modules:  # pragma: no cover - helper stub for headless CI
    pyqt6_stub = types.ModuleType("PyQt6")
    qtcore_stub = types.ModuleType("PyQt6.QtCore")

    class _QObject:  # pragma: no cover - placeholder QObject
        pass

    class _QRunnable:  # pragma: no cover - placeholder QRunnable
        pass

    class _QThreadPool:  # pragma: no cover - placeholder QThreadPool
        @staticmethod
        def globalInstance() -> "_QThreadPool":
            return _QThreadPool()

        def start(self, runnable: object) -> None:  # noqa: ARG002
            pass

    class _QTimer:  # pragma: no cover - placeholder QTimer
        def __init__(self) -> None:
            self._interval = 0

        def setInterval(self, interval: int) -> None:
            self._interval = interval

        def start(self) -> None:
            pass

    def _pyqtSignal(*_args: object, **_kwargs: object) -> object:
        return object()

    qtcore_stub.QObject = _QObject
    qtcore_stub.QRunnable = _QRunnable
    qtcore_stub.QThreadPool = _QThreadPool
    qtcore_stub.QTimer = _QTimer
    qtcore_stub.pyqtSignal = _pyqtSignal
    pyqt6_stub.QtCore = qtcore_stub
    sys.modules["PyQt6"] = pyqt6_stub
    sys.modules["PyQt6.QtCore"] = qtcore_stub

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

from core.pipeline import scan_and_tag
from db.connection import get_conn
from db.repository import replace_file_tags, upsert_file, upsert_tags
from utils import paths


@pytest.fixture()
def temp_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    
    class _StubPaths:
        def __init__(self, base: Path) -> None:
            self._base = base

        def data_dir(self) -> Path:
            return self._base

        def db_path(self) -> Path:
            return self._base / "kobato-eyes.db"

        def index_dir(self) -> Path:
            return self._base / "index"

        def cache_dir(self) -> Path:
            return self._base / "cache"

        def log_dir(self) -> Path:
            return self._base / "logs"

        def ensure_data_dirs(self) -> None:  # pragma: no cover - not used
            pass

        def migrate_data_dir_if_needed(self) -> bool:  # pragma: no cover - not used
            return False

    monkeypatch.setattr(paths, "_APP_PATHS", _StubPaths(data_dir))
    dummy_settings = types.SimpleNamespace(
        allow_exts={".png"},
        tagger=types.SimpleNamespace(thresholds={}, max_tags=None),
    )
    monkeypatch.setattr("core.pipeline.manual_refresh.load_settings", lambda: dummy_settings)
    return data_dir


def _make_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scan_and_tag_returns_empty_stats_for_missing_path(temp_env: Path) -> None:
    missing_root = temp_env / "not-there"

    stats = scan_and_tag(missing_root)

    assert stats == {
        "queued": 0,
        "tagged": 0,
        "elapsed_sec": 0.0,
        "missing": 0,
        "soft_deleted": 0,
        "hard_deleted": 0,
    }
    assert not paths.get_db_path().exists()


def test_scan_and_tag_skips_unsupported_extension(temp_env: Path, tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()
    unsupported = root / "note.txt"
    unsupported.write_text("hello", encoding="utf-8")

    stats = scan_and_tag(unsupported)

    assert int(stats.get("queued", -1)) == 0
    assert int(stats.get("tagged", -1)) == 0
    assert int(stats.get("missing", -1)) == 0
    assert int(stats.get("soft_deleted", -1)) == 0
    assert int(stats.get("hard_deleted", -1)) == 0
    assert stats.get("elapsed_sec") is not None
    assert stats.get("elapsed_sec", -1.0) >= 0.0
    assert not paths.get_db_path().exists()


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


def test_scan_and_tag_avoids_duplicate_queue(
    monkeypatch: pytest.MonkeyPatch, temp_env: Path, tmp_path: Path
) -> None:
    import core.pipeline.manual_refresh as manual_refresh

    root = tmp_path / "library"
    root.mkdir()
    candidate = root / "dup.png"
    candidate.write_bytes(b"data")

    db_path = paths.get_db_path()
    conn = get_conn(db_path)
    try:
        upsert_file(
            conn,
            path=str(candidate.resolve()),
            size=candidate.stat().st_size,
            mtime=candidate.stat().st_mtime,
            sha256=_make_sha(candidate),
        )
    finally:
        conn.close()

    calls: list[Path] = []

    class _DummyTagger:
        def close(self) -> None:
            calls.append(Path("<closed>"))

    monkeypatch.setattr(manual_refresh, "_resolve_tagger", lambda *args, **kwargs: _DummyTagger())

    def _fake_run_tag_job(tagger: object, path_obj: Path, conn: object, *, config: object) -> None:
        calls.append(Path(path_obj))
        return None

    monkeypatch.setattr(manual_refresh, "run_tag_job", _fake_run_tag_job)

    stats = scan_and_tag(root)

    queued = int(stats.get("queued", -1))
    assert queued == 1
    assert [p for p in calls if p != Path("<closed>")] == [candidate.resolve()]
