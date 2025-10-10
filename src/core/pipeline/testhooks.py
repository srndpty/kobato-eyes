from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from core.db_writer import DBWriter
from core.pipeline.contracts import DBWriteQueue

from .loaders import PrefetchLoaderPrepared

# ---- Loader / DBWriter / Quiesce の注入ポイント ----
LoaderFactory = Callable[
    [List[str], object, int, int, Optional[int]], Iterable[Tuple[list[str], np.ndarray, list[tuple[int, int]]]]
]
DBWriterFactory = Callable[..., "IDBWriterLike"]


class IQuiesceCtrl:
    def begin(self) -> None: ...
    def end(self) -> None: ...


class _DefaultQuiesceCtrl(IQuiesceCtrl):
    def begin(self) -> None:
        from db.connection import begin_quiesce

        begin_quiesce()

    def end(self) -> None:
        from db.connection import end_quiesce

        end_quiesce()


def default_loader_factory(paths: List[str], tagger, batch_size: int, prefetch_batches: int, io_workers: Optional[int]):
    return PrefetchLoaderPrepared(
        paths,
        tagger=tagger,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        io_workers=io_workers,
    )


class IDBWriterLike(DBWriteQueue):
    """Backwards compatible alias for DB writing queues."""


def default_dbwriter_factory(
    *,
    db_path: str,
    flush_chunk: int,
    fts_topk: int,
    queue_size: int,
    default_tagger_sig: str,
    unsafe_fast: bool,
    skip_fts: bool,
    progress_cb,
):
    # 既存 DBWriter をそのまま返す（実運用用）
    return DBWriter(
        db_path,
        flush_chunk=flush_chunk,
        fts_topk=fts_topk,
        queue_size=queue_size,
        default_tagger_sig=default_tagger_sig,
        unsafe_fast=unsafe_fast,
        skip_fts=skip_fts,
        progress_cb=progress_cb,
    )


def _default_conn_factory(db_path: str):
    from db.connection import get_conn

    return get_conn(db_path, allow_when_quiesced=True)


@dataclass
class TaggingDeps:
    """TaggingStage のテスト容易性のための依存集合（デフォルトは実運用実装）。"""

    loader_factory: LoaderFactory = default_loader_factory
    dbwriter_factory: DBWriterFactory = default_dbwriter_factory
    quiesce: IQuiesceCtrl = _DefaultQuiesceCtrl()
    conn_factory: Callable[[str], object] = _default_conn_factory


__all__ = ["TaggingDeps", "IQuiesceCtrl", "IDBWriterLike", "default_loader_factory", "default_dbwriter_factory"]
