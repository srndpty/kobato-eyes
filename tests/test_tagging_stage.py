from __future__ import annotations

import time

import numpy as np

from core.pipeline.stages.tag_stage import TagStage
from core.pipeline.stages.write_stage import WriteStage
from core.pipeline.testhooks import IDBWriterLike, IQuiesceCtrl, TaggingDeps
from core.pipeline.types import IndexPhase, PipelineContext, ProgressEmitter, _FileRecord
from core.settings import PipelineSettings
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
        self.started = False

    def start(self):
        self.started = True

    def raise_if_failed(self):
        pass

    def put(self, item):
        self.items.append(item)
        self._q += 1

    def qsize(self):
        return self._q

    def stop(self, *, flush: bool, wait_forever: bool):
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
        conn_factory=lambda _: type("C", (), {"close": lambda self: None})(),
    )
    ctx = _ctx()
    tag_stage = TagStage(deps=deps)
    write_stage = WriteStage(
        deps=type(
            "Deps",
            (),
            {
                "build_writer": lambda self, ctx, progress_cb: fake_db,
                "begin_quiesce": lambda self: None,
                "end_quiesce": lambda self: None,
                "connect": lambda self, db_path: None,
            },
        )()
    )

    tag_result = tag_stage.run(ctx, emitter, recs)
    write_result = write_stage.run(ctx, emitter, tag_result)

    assert tag_result.tagged_count == 2
    assert len(fake_db.items) == 2
    # DBItem の中身ざっくり（ファイルIDが入っている）
    ids = sorted(it.file_id for it in fake_db.items)
    assert ids == [1, 2]
    # 進捗が少なくとも TAG フェーズで1回以上通知される
    assert any(ph == IndexPhase.TAG for (ph, _, _) in prog)
    # 書き込みフェーズで FTS 進捗が通知される
    assert any(ph == IndexPhase.FTS for (ph, _, _) in prog)
    assert write_result.written == 2
