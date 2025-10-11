"""ProgressEmitter の挙動に関するテスト。"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

MODULE_NAME = "core.pipeline.types"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "src" / "core" / "pipeline" / "types.py"


def _load_progress_module() -> ModuleType:
    if MODULE_NAME in sys.modules:
        module = sys.modules[MODULE_NAME]
        assert isinstance(module, ModuleType)
        return module

    core_module = ModuleType("core")
    core_module.__path__ = []  # type: ignore[attr-defined]

    config_module = ModuleType("core.config")

    class PipelineSettings:  # noqa: D401 - ドキュメント不要の簡易スタブ
        """設定スタブ。"""

    config_module.PipelineSettings = PipelineSettings  # type: ignore[attr-defined]
    core_module.config = config_module  # type: ignore[attr-defined]

    sys.modules.setdefault("core", core_module)
    sys.modules.setdefault("core.config", config_module)

    tagger_module = ModuleType("tagger")
    tagger_module.__path__ = []  # type: ignore[attr-defined]
    tagger_base_module = ModuleType("tagger.base")

    class ITagger:  # noqa: D401 - ドキュメント不要の簡易スタブ
        """タガースタブ。"""

    class TagCategory:  # noqa: D401 - ドキュメント不要の簡易スタブ
        """カテゴリー列挙スタブ。"""

    tagger_base_module.ITagger = ITagger  # type: ignore[attr-defined]
    tagger_base_module.TagCategory = TagCategory  # type: ignore[attr-defined]
    tagger_module.base = tagger_base_module  # type: ignore[attr-defined]

    sys.modules.setdefault("tagger", tagger_module)
    sys.modules.setdefault("tagger.base", tagger_base_module)

    pil_module = ModuleType("PIL")

    class _Image:  # noqa: D401 - ドキュメント不要の簡易スタブ
        """Pillow Image スタブ。"""

    pil_module.Image = _Image  # type: ignore[attr-defined]
    sys.modules.setdefault("PIL", pil_module)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load progress module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _types_module() -> ModuleType:
    return _load_progress_module()


def _progress_objects() -> tuple[type, type, type]:
    module = _types_module()
    IndexPhase = getattr(module, "IndexPhase")
    IndexProgress = getattr(module, "IndexProgress")
    ProgressEmitter = getattr(module, "ProgressEmitter")
    return IndexPhase, IndexProgress, ProgressEmitter


IndexPhase, IndexProgress, ProgressEmitter = _progress_objects()


def test_emit_disables_callback_after_exception(caplog: pytest.LogCaptureFixture) -> None:
    """一度例外が発生したら以降の emit が無視されることを検証する。"""

    call_count = 0

    def fake_callback(progress: IndexProgress) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    emitter = ProgressEmitter(fake_callback)
    progress = IndexProgress(phase=IndexPhase.SCAN, done=1, total=10)

    with caplog.at_level(logging.ERROR):
        emitter.emit(progress, force=True)
        emitter.emit(progress, force=True)

    assert call_count == 1
    assert emitter._cb is None  # type: ignore[attr-defined]
    assert any(
        "Progress callback raised" in record.getMessage() for record in caplog.records
    )


def test_cancelled_logs_exception(caplog: pytest.LogCaptureFixture) -> None:
    """キャンセル判定の例外が記録され False が返ることを検証する。"""

    def fake_cancelled() -> bool:
        raise RuntimeError("cancel boom")

    emitter = ProgressEmitter(lambda _: None)

    with caplog.at_level(logging.ERROR):
        result = emitter.cancelled(fake_cancelled)

    assert result is False
    assert any(
        "Cancellation callback failed" in record.getMessage()
        and record.name == MODULE_NAME
        for record in caplog.records
    )
