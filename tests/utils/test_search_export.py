"""Tests for utils.search_export helpers."""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pytest

# ``utils.search_export`` depends on ``core.config`` at import time, which in
# turn pulls in heavy third-party libraries. For these focused unit tests we
# provide lightweight stubs so that the module can be imported without optional
# dependencies being installed.
if "core" not in sys.modules:  # pragma: no cover - defensive guard
    core_stub = types.ModuleType("core")
    config_stub = types.ModuleType("core.config")

    class _StubAppPaths:
        def __init__(self) -> None:
            self._base = Path.cwd() / "_stub_app_paths"

        def data_dir(self) -> Path:  # pragma: no cover - stub behaviour
            return self._base / "data"

        def db_path(self) -> Path:  # pragma: no cover - stub behaviour
            return self.data_dir() / "kobato-eyes.db"

        def index_dir(self) -> Path:  # pragma: no cover - stub behaviour
            return self.data_dir() / "index"

        def cache_dir(self) -> Path:  # pragma: no cover - stub behaviour
            return self.data_dir() / "cache"

        def log_dir(self) -> Path:  # pragma: no cover - stub behaviour
            return self.data_dir() / "logs"

        def ensure_data_dirs(self) -> None:  # pragma: no cover - stub behaviour
            pass

        def migrate_data_dir_if_needed(self) -> bool:  # pragma: no cover - stub behaviour
            return False

    config_stub.AppPaths = _StubAppPaths
    config_stub.PipelineSettings = object
    config_stub.TaggerSettings = object
    config_stub.SettingsService = object
    config_stub.configure = lambda app_paths: None  # pragma: no cover - stub
    config_stub.config_path = lambda: Path.cwd() / "config.yaml"  # pragma: no cover - stub
    config_stub.load_settings = lambda: None  # pragma: no cover - stub
    config_stub.save_settings = lambda settings: None  # pragma: no cover - stub
    config_stub.__all__ = [
        "AppPaths",
        "PipelineSettings",
        "SettingsService",
        "TaggerSettings",
        "config_path",
        "configure",
        "load_settings",
        "save_settings",
    ]

    core_stub.config = config_stub
    sys.modules["core"] = core_stub
    sys.modules["core.config"] = config_stub

# Stub minimal interfaces for third-party libraries that are not available in
# the execution environment.
if "pydantic" not in sys.modules:  # pragma: no cover - defensive guard
    stub = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **fields: Any) -> None:
            for key, value in fields.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: Any) -> "_BaseModel":
            if isinstance(data, dict):
                return cls(**data)
            return cls()

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

    def _config_dict(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    def _field(*args: Any, default: Any = None, default_factory: Callable[[], Any] | None = None, **kwargs: Any) -> Any:
        if default_factory is not None:
            return default_factory()
        return default

    def _passthrough_decorator(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    stub.BaseModel = _BaseModel
    stub.ConfigDict = _config_dict
    stub.Field = _field
    stub.field_validator = _passthrough_decorator
    stub.model_validator = _passthrough_decorator
    sys.modules["pydantic"] = stub

if "yaml" not in sys.modules:  # pragma: no cover - defensive guard
    yaml_stub = types.ModuleType("yaml")

    def _not_available(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - stub
        raise RuntimeError("yaml stub is only for import-time usage in tests")

    yaml_stub.safe_load = _not_available
    yaml_stub.safe_dump = _not_available
    sys.modules["yaml"] = yaml_stub

from utils import search_export


@pytest.mark.parametrize(
    ("raw", "expected", "max_len"),
    [
        ("simple", "simple", None),
        ("dog/cat:mouse", "dog_cat_mouse", None),
        ("  spaced   out name  ", "spaced_out_name", None),
        ("MiXeD\tSpaces\nand\r\nlines", "MiXeD_Spaces_and_lines", None),
        ("??", "_", None),
        ("", "query", None),
        ("   ", "query", None),
        ("a" * 100, "a" * 10, 10),
    ],
)
def test_sanitize_for_folder_cases(raw: str, expected: str, max_len: int | None) -> None:
    params: dict[str, int] = {}
    if max_len is not None:
        params["max_len"] = max_len

    assert search_export.sanitize_for_folder(raw, **params) == expected


def test_make_export_dir_creates_expected_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDatetime:
        @classmethod
        def now(cls) -> datetime:
            return datetime(2025, 1, 3, 14, 22, 33)

    monkeypatch.setattr(search_export, "datetime", FixedDatetime)
    monkeypatch.setattr(search_export, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(search_export, "ensure_dirs", lambda: None)

    export_dir = search_export.make_export_dir("Dog cat??  ")

    expected_root = tmp_path / "search_results"
    expected_name = "20250103-142233-Dog_cat_"

    assert export_dir == expected_root / expected_name
    assert export_dir.is_dir()
    assert expected_root.is_dir()
