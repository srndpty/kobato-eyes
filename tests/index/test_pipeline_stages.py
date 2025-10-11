from __future__ import annotations

from pathlib import Path

from core.pipeline.orchestrator import IndexPipeline
from core.pipeline.stages.scan_stage import ScanStageResult
from core.pipeline.stages.tag_stage import TagStageResult
from core.pipeline.stages.write_stage import WriteStageResult
from core.pipeline.types import PipelineContext, _FileRecord


class FakeScanStage:
    def __init__(self) -> None:
        self.called = False

    def run(self, ctx: PipelineContext, emitter, *_):
        self.called = True
        record = _FileRecord(
            file_id=1,
            path=Path("/tmp/a.jpg"),
            size=1,
            mtime=0.0,
            sha="x",
            is_new=True,
            changed=False,
            tag_exists=False,
            needs_tagging=True,
        )
        return ScanStageResult(records=[record], scanned=1, new_or_changed=1)


class FakeTagStage:
    def __init__(self) -> None:
        self.received: list[_FileRecord] | None = None
        self.called = False

    def run(self, ctx: PipelineContext, emitter, records):
        self.called = True
        self.received = records
        return TagStageResult(records=records, db_items=[], tagged_count=5)


class FakeWriteStage:
    def __init__(self) -> None:
        self.received = None
        self.called = False

    def run(self, ctx: PipelineContext, emitter, tag_result):
        self.called = True
        self.received = tag_result
        return WriteStageResult(written=3, fts_processed=3)


def test_index_pipeline_allows_stage_overrides(tmp_path):
    pipeline = IndexPipeline(db_path=tmp_path / "test.db")
    fake_scan = FakeScanStage()
    fake_tag = FakeTagStage()
    fake_write = FakeWriteStage()

    pipeline.set_stage_override("scan", fake_scan)
    pipeline.set_stage_override("tag", fake_tag)
    pipeline.set_stage_override("write", fake_write)

    stats = pipeline.run()

    assert fake_scan.called
    assert fake_tag.received is not None
    assert fake_write.received is not None
    assert stats["scanned"] == 1
    assert stats["new_or_changed"] == 1
    assert stats["tagged"] == 5
    assert stats["signatures"] == 3


def test_index_pipeline_stops_on_cancellation(tmp_path):
    pipeline = IndexPipeline(db_path=tmp_path / "test.db", is_cancelled=lambda: True)
    fake_scan = FakeScanStage()
    fake_tag = FakeTagStage()
    fake_write = FakeWriteStage()

    pipeline.set_stage_override("scan", fake_scan)
    pipeline.set_stage_override("tag", fake_tag)
    pipeline.set_stage_override("write", fake_write)

    stats = pipeline.run()

    assert fake_scan.called
    assert not fake_tag.called
    assert not fake_write.called
    assert stats["tagged"] == 0
    assert stats["signatures"] == 0
    assert stats["cancelled"]
