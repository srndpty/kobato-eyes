"""Hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 digest for the file at ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["compute_sha256"]
