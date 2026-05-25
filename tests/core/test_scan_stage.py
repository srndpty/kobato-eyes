"""Tests for scan stage database lookup behaviour."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import PipelineSettings
from core.pipeline.stages import scan_stage
from core.pipeline.stages.scan_stage import ScanStage
from core.pipeline.types import PipelineContext, ProgressEmitter


class _Conn:
    """Minimal connection double for scan-stage tests."""

    def commit(self) -> None:
        """Record a commit; no-op for the test double."""

    def close(self) -> None:
        """Close the connection; no-op for the test double."""


class _BulkEmptyDeps:
    """Database dependencies where bulk fetch succeeds with no existing rows."""

    def __init__(self) -> None:
        self.conn = _Conn()
        self.fetch_file_calls = 0
        self.bulk_fetch_paths: list[str] = []

    def get_connection(self, db_path: str) -> _Conn:
        """Return the fake connection."""

        return self.conn

    def fetch_file(self, conn: _Conn, path: str) -> Any:
        """Track legacy per-file lookups."""

        self.fetch_file_calls += 1
        return None

    def fetch_files_by_path(self, conn: _Conn, paths: list[str]) -> dict[str, object]:
        """Return an empty bulk result, as on a first scan."""

        self.bulk_fetch_paths = list(paths)
        return {}

    def upsert_file(
        self,
        conn: _Conn,
        *,
        path: str,
        size: int,
        mtime: float,
        sha256: str,
        indexed_at: float | None,
    ) -> int:
        """Pretend the file was inserted."""

        return 123

    def has_tag(self, conn: _Conn, file_id: int) -> bool:
        """New files have no tags."""

        return False


def test_scan_stage_does_not_fallback_to_fetch_file_after_empty_bulk_fetch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not-a-real-image-but-hashable")
    deps = _BulkEmptyDeps()
    monkeypatch.setattr(scan_stage, "iter_images", lambda *args, **kwargs: iter([image_path]))

    ctx = PipelineContext(
        db_path=tmp_path / "index.db",
        settings=PipelineSettings(roots=[str(tmp_path)], allow_exts={".jpg"}),
        thresholds={},
        max_tags_map={},
        tagger_sig="sig",
    )

    result = ScanStage(deps).run(ctx, ProgressEmitter(None))

    assert result.scanned == 1
    assert result.new_or_changed == 1
    assert deps.bulk_fetch_paths == [str(image_path)]
    assert deps.fetch_file_calls == 0


def test_scan_stage_keeps_iteration_streaming_for_cancellation(monkeypatch, tmp_path: Path) -> None:
    paths = []
    for index in range(5):
        image_path = tmp_path / f"sample-{index}.jpg"
        image_path.write_bytes(b"hashable")
        paths.append(image_path)
    yielded: list[Path] = []

    def _iter_images(*args, **kwargs):
        for image_path in paths:
            yielded.append(image_path)
            yield image_path

    cancelled = False

    def _on_progress(_progress) -> None:
        nonlocal cancelled
        cancelled = True

    monkeypatch.setattr(scan_stage, "iter_images", _iter_images)

    ctx = PipelineContext(
        db_path=tmp_path / "index.db",
        settings=PipelineSettings(roots=[str(tmp_path)], allow_exts={".jpg"}),
        thresholds={},
        max_tags_map={},
        tagger_sig="sig",
        is_cancelled=lambda: cancelled,
    )

    result = ScanStage(_BulkEmptyDeps()).run(ctx, ProgressEmitter(_on_progress))

    assert result.scanned == 1
    assert len(yielded) < len(paths)
