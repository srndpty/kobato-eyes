from __future__ import annotations

import time

import numpy as np

from core.pipeline.tagging import TaggingStage
from core.pipeline.testhooks import IDBWriterLike, IQuiesceCtrl, TaggingDeps
from core.pipeline.types import IndexPhase, PipelineContext, ProgressEmitter, _FileRecord
from core.config import PipelineSettings
from tagger.base import ITagger, TagCategory, TagPrediction, TagResult


# ---------- Fakes ----------
class FakeTagger(ITagger):
    def prepare_batch_from_bgr(self, imgs):
        # (B, 448, 448, 3) float32 風の配列だけ用意
        B = len(imgs)
        return np.zeros((B, 448, 448, 3), dtype=np.float32)

    def prepare_batch_from_rgb_np(self, imgs):
        return self.prepare_batch_from_bgr(imgs)

    def infer_batch_prepared(self, np_batch, thresholds=None, max_tags=None):
        B = np_batch.shape[0]
        res = []
        for _ in range(B):
            res.append(TagResult(tags=[TagPrediction(name="unit_test_tag", score=0.99, category=TagCategory.GENERAL)]))
        return res


class FakeLoader:
    def __init__(self, paths, *_, **__):
        self._paths = list(paths)
        self._done = False

    def __iter__(self):
        if self._done:
            return iter(())
        self._done = True
        B = len(self._paths)
        batch = np.zeros((B, 448, 448, 3), dtype=np.float32)
        sizes = [(448, 448)] * B
        yield (self._paths, batch, sizes)

    def close(self):
        pass


class FakeDBWriter(IDBWriterLike):
    def __init__(self):
        self.items = []
        self._q = 0

    def start(self):
        pass

    def raise_if_failed(self):
        pass

    def put(self, item):
        self.items.append(item)
        self._q += 1

    def qsize(self):
        return self._q

    def stop(self, *, flush: bool, wait_forever: bool):
        pass


class FakeWriteDeps:
    def __init__(self, writer: IDBWriterLike) -> None:
        self._writer = writer

    def build_writer(self, *, ctx, progress_cb):
        return self._writer

    def begin_quiesce(self) -> None:
        pass

    def end_quiesce(self) -> None:
        pass

    def connect(self, db_path: str):
        pass


class NoopQuiesce(IQuiesceCtrl):
    def begin(self):
        pass

    def end(self):
        pass


def _ctx():
    s = PipelineSettings()
    s.tagger.name = "wd14-onnx"
    s.tagger.thresholds = {"general": 0.35}
    return PipelineContext(
        db_path=":memory:",
        settings=s,
        thresholds={TagCategory.GENERAL: 0.35},
        max_tags_map={},
        tagger_sig="unittest:sig",
        tagger_override=FakeTagger(),
        progress_cb=None,
        is_cancelled=None,
    )


def test_tagging_stage_with_fakes():
    # records: 2件、どちらもタグ付け対象
    recs = [
        _FileRecord(
            file_id=1,
            path="a.jpg",
            size=1,
            mtime=time.time(),
            sha="x",
            is_new=True,
            changed=False,
            tag_exists=False,
            needs_tagging=True,
        ),
        _FileRecord(
            file_id=2,
            path="b.jpg",
            size=1,
            mtime=time.time(),
            sha="y",
            is_new=True,
            changed=False,
            tag_exists=False,
            needs_tagging=True,
        ),
    ]
    prog = []
    emitter = ProgressEmitter(lambda p: prog.append((p.phase, p.done, p.total)))
    fake_db = FakeDBWriter()
    deps = TaggingDeps(
        loader_factory=lambda paths, tagger, B, depth, io: FakeLoader(paths),
        dbwriter_factory=lambda **kw: fake_db,
        quiesce=NoopQuiesce(),
    )
    stage = TaggingStage(_ctx(), emitter, deps=deps, writer_deps=FakeWriteDeps(fake_db))
    tagged, fts, _ = stage.run(recs)

    assert tagged == 2
    assert len(fake_db.items) == 2
    # DBItem の中身ざっくり（ファイルIDが入っている）
    ids = sorted(it.file_id for it in fake_db.items)
    assert ids == [1, 2]
    # 進捗が少なくとも TAG フェーズで1回以上通知される
    assert any(ph == IndexPhase.TAG for (ph, _, _) in prog)
