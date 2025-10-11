"""Tests for the tagging pipeline stage."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

pytest.importorskip("pydantic")
np = pytest.importorskip("numpy")

from core.config.schema import PipelineSettings
from core.pipeline.stages.tag_stage import LoaderIterable, TagStage, TagStageDeps
from core.pipeline.types import IndexProgress, PipelineContext, ProgressEmitter, _FileRecord
from tagger.base import TagCategory, TagPrediction, TagResult


class FakeTagger:
    """Tagger that fails for prepared batches larger than one sample."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def infer_batch_prepared(
        self,
        batch: np.ndarray,
        *,
        thresholds: dict[TagCategory, float] | None = None,
        max_tags: dict[TagCategory, int] | None = None,
    ) -> list[TagResult]:
        del thresholds, max_tags
        size = int(batch.shape[0])
        self.batch_sizes.append(size)
        if size > 1:
            raise RuntimeError("batch too large")
        identifier = int(batch[0, 0, 0, 0])
        return [
            TagResult(
                tags=[TagPrediction(name=f"tag-{identifier}", score=0.9, category=TagCategory.GENERAL)]
            )
        ]


class SingleBatchLoader:
    """Loader that yields a single prepared batch and records when it is closed."""

    def __init__(self, payload: tuple[list[str], np.ndarray, list[tuple[int, int]]]) -> None:
        self._payload = payload
        self.closed = False

    def __iter__(self) -> Iterable[tuple[list[str], np.ndarray, list[tuple[int, int]]]]:
        yield self._payload

    def close(self) -> None:
        self.closed = True

    def qsize(self) -> int:
        return 0


class FakeDeps(TagStageDeps):
    """Dependency provider returning a predetermined loader."""

    def __init__(self, loader_factory: Callable[[list[str]], LoaderIterable]) -> None:
        self._factory = loader_factory
        self.created_loader: SingleBatchLoader | None = None

    def loader_factory(
        self,
        paths: list[str],
        tagger,
        batch_size: int,
        prefetch_batches: int,
        io_workers: int | None,
    ) -> LoaderIterable:
        del tagger, batch_size, prefetch_batches, io_workers
        loader = self._factory(paths)
        assert isinstance(loader, SingleBatchLoader)
        self.created_loader = loader
        return loader


def _make_record(path: Path, file_id: int) -> _FileRecord:
    return _FileRecord(
        file_id=file_id,
        path=path,
        size=100,
        mtime=0.0,
        sha=f"sha-{file_id}",
        is_new=True,
        changed=False,
        tag_exists=False,
        needs_tagging=True,
    )


@pytest.mark.parametrize("count", [3])
def test_tag_stage_recovers_from_split_batches(tmp_path: Path, count: int) -> None:
    fake_tagger = FakeTagger()
    records = [_make_record(tmp_path / f"img_{idx}.png", idx) for idx in range(count)]

    image_map: dict[str, np.ndarray] = {
        str(record.path): np.full((2, 2, 3), fill_value=record.file_id, dtype=np.float32)
        for record in records
    }

    def factory(paths: list[str]) -> SingleBatchLoader:
        arrays = [image_map[path] for path in paths]
        prepared = np.stack(arrays, axis=0)
        sizes = [(arr.shape[1], arr.shape[0]) for arr in arrays]
        return SingleBatchLoader((paths, prepared, sizes))

    deps = FakeDeps(factory)
    stage = TagStage(deps=deps)

    progress_events: list[IndexProgress] = []

    def progress_cb(progress: IndexProgress) -> None:
        progress_events.append(progress)

    emitter = ProgressEmitter(progress_cb)
    ctx = PipelineContext(
        db_path=tmp_path / "db.sqlite3",
        settings=PipelineSettings(),
        thresholds={},
        max_tags_map={},
        tagger_sig="sig",
        tagger_override=fake_tagger,
        progress_cb=progress_cb,
        is_cancelled=None,
    )

    result = stage.run(ctx, emitter, records)

    assert result.tagged_count == count
    assert len(result.db_items) == count
    assert deps.created_loader is not None and deps.created_loader.closed
    assert fake_tagger.batch_sizes[0] == count
    assert sorted(fake_tagger.batch_sizes[1:]) == [1] * count

    expected_tags = {f"tag-{record.file_id}" for record in records}
    produced_tags = {name for item in result.db_items for name, _score, _cat in item.tags}
    assert produced_tags == expected_tags

    assert all(not record.needs_tagging for record in records)
    assert all(record.tag_exists for record in records)

    assert progress_events
    assert progress_events[0].done == 0 and progress_events[0].total == count
    assert any(event.done == count and event.total == count for event in progress_events)
