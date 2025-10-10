"""Compatibility wrapper exposing the DB writing service."""

from __future__ import annotations

from core.pipeline.contracts import DBFlush, DBItem, DBStop, DBWriteQueue
from services.db_writing import DBWritingService


class DBWriter(DBWritingService):
    """Alias kept for historical imports within the core package."""


__all__ = ["DBWriter", "DBItem", "DBFlush", "DBStop", "DBWriteQueue"]
