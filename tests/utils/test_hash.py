"""Tests for :mod:`utils.hash`."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from utils.hash import compute_sha256


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    """Ensure :func:`compute_sha256` matches :mod:`hashlib` results."""
    tmp_file = tmp_path / "large.bin"
    tmp_file.write_bytes(os.urandom(5 * 1024 * 1024))

    expected = hashlib.sha256(tmp_file.read_bytes()).hexdigest()

    assert compute_sha256(tmp_file) == expected
    assert compute_sha256(tmp_file, chunk_size=1) == expected
