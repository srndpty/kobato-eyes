"""WriteStage におけるエラー時クリーンアップの挙動を検証するテスト。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.pipeline.stages import write_stage
from core.pipeline.stages.tag_stage import TagStageResult
from core.pipeline.stages.write_stage import WriteStage
from core.pipeline.types import PipelineContext, ProgressEmitter


def _make_ctx(tmp_path) -> PipelineContext:
    settings = SimpleNamespace(db_flush_chunk=16, fts_topk=32)
    return PipelineContext(
        db_path=tmp_path / "test.db",
        settings=settings,
        thresholds={},
        max_tags_map={},
        tagger_sig="sig",
        tagger_override=None,
        progress_cb=None,
        is_cancelled=lambda: False,
    )


def test_write_stage_cleans_up_after_writer_failure(monkeypatch, tmp_path):
    settle_calls: list[str] = []

    monkeypatch.setattr(
        write_stage,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(str(path)),
    )

    class _FailingWriter:
        def __init__(self) -> None:
            self.stop_calls = 0

        def start(self) -> None:
            raise RuntimeError("writer failed during start")

        def raise_if_failed(self) -> None:  # pragma: no cover - never reached
            pytest.fail("raise_if_failed should not be called after start failure")

        def put(self, _item) -> None:  # pragma: no cover - never reached
            pytest.fail("put should not be called when start fails")

        def stop(self, *, flush: bool, wait_forever: bool) -> None:
            assert flush is True
            assert wait_forever is True
            self.stop_calls += 1

    class _Deps:
        def __init__(self) -> None:
            self.begin_calls = 0
            self.end_calls = 0
            self.writer = _FailingWriter()

        def connect(self, db_path: str) -> None:
            assert db_path

        def begin_quiesce(self) -> None:
            self.begin_calls += 1

        def end_quiesce(self) -> None:
            self.end_calls += 1

        def build_writer(self, *, ctx: PipelineContext, progress_cb):
            assert ctx
            assert callable(progress_cb)
            return self.writer

        def rebuild_fts(self, db_path: str, *, topk: int) -> int:
            pytest.fail("FTS rebuild should not run when writer start fails")

    deps = _Deps()
    stage = WriteStage(deps=deps)
    ctx = _make_ctx(tmp_path)
    emitter = ProgressEmitter(lambda _progress: None)
    tag_result = TagStageResult(records=[], db_items=[object()], tagged_count=1)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 0
    assert result.fts_processed == 0
    assert result.success is False
    assert result.error == "writer failed during start"
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert deps.writer.stop_calls == 1
    assert settle_calls == [str(ctx.db_path)]


def test_write_stage_cleans_up_when_build_writer_fails(monkeypatch, tmp_path):
    settle_calls: list[str] = []

    monkeypatch.setattr(
        write_stage,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(str(path)),
    )

    class _Deps:
        def __init__(self) -> None:
            self.begin_calls = 0
            self.end_calls = 0
            self.connect_calls = 0

        def connect(self, db_path: str) -> None:
            assert db_path
            self.connect_calls += 1

        def begin_quiesce(self) -> None:
            self.begin_calls += 1

        def end_quiesce(self) -> None:
            self.end_calls += 1

        def build_writer(self, *, ctx: PipelineContext, progress_cb):
            assert ctx
            assert callable(progress_cb)
            raise RuntimeError("failed to build writer")

        def rebuild_fts(self, db_path: str, *, topk: int) -> int:
            pytest.fail("FTS rebuild should not run when writer creation fails")

    deps = _Deps()
    stage = WriteStage(deps=deps)
    ctx = _make_ctx(tmp_path)
    emitter = ProgressEmitter(lambda _progress: None)
    tag_result = TagStageResult(records=[], db_items=[object()], tagged_count=1)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 0
    assert result.fts_processed == 0
    assert result.success is False
    assert result.error == "failed to build writer"
    assert deps.connect_calls == 1
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert settle_calls == [str(ctx.db_path)]


def test_write_stage_rebuilds_fts_after_successful_fast_write(monkeypatch, tmp_path):
    settle_calls: list[str] = []

    monkeypatch.setattr(
        write_stage,
        "_settle_after_quiesce",
        lambda path: settle_calls.append(str(path)),
    )

    class _Writer:
        def __init__(self) -> None:
            self.started = False
            self.items: list[object] = []
            self.stop_calls = 0

        def start(self) -> None:
            self.started = True

        def raise_if_failed(self) -> None:
            assert self.started

        def put(self, item) -> None:
            self.items.append(item)

        def stop(self, *, flush: bool, wait_forever: bool) -> None:
            assert flush is True
            assert wait_forever is True
            self.stop_calls += 1

    class _Deps:
        def __init__(self) -> None:
            self.begin_calls = 0
            self.end_calls = 0
            self.writer = _Writer()
            self.rebuild_calls: list[tuple[str, int]] = []

        def connect(self, db_path: str) -> None:
            assert db_path

        def begin_quiesce(self) -> None:
            self.begin_calls += 1

        def end_quiesce(self) -> None:
            self.end_calls += 1

        def build_writer(self, *, ctx: PipelineContext, progress_cb):
            assert ctx
            assert callable(progress_cb)
            return self.writer

        def rebuild_fts(self, db_path: str, *, topk: int) -> int:
            self.rebuild_calls.append((db_path, topk))
            return 1

    deps = _Deps()
    stage = WriteStage(deps=deps)
    ctx = _make_ctx(tmp_path)
    emitter = ProgressEmitter(lambda _progress: None)
    item = object()
    tag_result = TagStageResult(records=[], db_items=[item], tagged_count=1)

    result = stage.run(ctx, emitter, tag_result)

    assert result.written == 1
    assert result.fts_processed == 1
    assert result.success is True
    assert result.error is None
    assert deps.writer.items == [item]
    assert deps.writer.stop_calls == 1
    assert deps.begin_calls == 1
    assert deps.end_calls == 1
    assert deps.rebuild_calls == [(str(ctx.db_path), 32)]
    assert settle_calls == [str(ctx.db_path), str(ctx.db_path)]
