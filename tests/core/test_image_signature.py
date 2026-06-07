"""Regression tests for inline image-signature storage (core.signature)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from PIL import Image

from core.signature import _to_signed64, compute_signatures_from_image, ensure_signatures
from dup.scanner import _parse_phash_any


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE signatures (file_id INTEGER PRIMARY KEY, phash_u64 INTEGER NOT NULL, dhash_u64 INTEGER NOT NULL)"
    )
    return conn


def _random_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
    return Image.fromarray(arr)


def test_signatures_are_signed_64bit() -> None:
    """格納値は SQLite が扱える符号付き64bit範囲に収まらなければならない。"""
    for seed in range(20):
        p, d = compute_signatures_from_image(_random_image(seed))
        for value in (p, d):
            assert -(1 << 63) <= value <= (1 << 63) - 1


def test_ensure_signatures_persists_without_overflow() -> None:
    """以前は phash 最上位ビットが立つ画像で OverflowError → 無言失敗していた回帰。"""
    conn = _make_conn()
    stored = 0
    for file_id in range(1, 21):
        assert ensure_signatures(conn, file_id, image=_random_image(file_id)) is True
        stored += 1
    assert conn.execute("SELECT COUNT(*) FROM signatures").fetchone()[0] == stored


def test_signature_round_trips_to_unsigned() -> None:
    """符号付きで格納しても、重複スキャナ側のマスク復元で符号なし値に戻る。"""
    conn = _make_conn()
    image = _random_image(123)
    assert ensure_signatures(conn, 1, image=image) is True
    row = conn.execute("SELECT phash_u64 FROM signatures WHERE file_id = 1").fetchone()
    expected = _to_signed64(compute_signatures_from_image(image)[0]) & ((1 << 64) - 1)
    assert _parse_phash_any(row["phash_u64"]) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        ((1 << 64) - 1, -1),
        (1 << 63, -(1 << 63)),
        ((1 << 63) - 1, (1 << 63) - 1),
    ],
)
def test_to_signed64(value: int, expected: int) -> None:
    assert _to_signed64(value) == expected
