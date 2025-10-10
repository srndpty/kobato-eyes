"""View model supporting duplicate detection UI components."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

from PyQt6.QtCore import QObject

from db.connection import get_conn as _get_conn
from db.repository import iter_files_for_dup as _iter_files_for_dup, mark_files_absent as _mark_files_absent
from dup.scanner import DuplicateScanConfig, DuplicateScanner
from utils.image_io import generate_thumbnail as _generate_thumbnail, get_thumbnail as _get_thumbnail
from utils.paths import get_cache_dir as _get_cache_dir, get_db_path as _get_db_path


class DupViewModel(QObject):
    """Provide data operations for :class:`ui.dup_tab.DupTab`."""

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        db_path: Path | None = None,
        connection_factory: Callable[[Path], object] = _get_conn,
        iter_files_for_dup: Callable[[object, str | None], Iterable[Sequence[object]]] = _iter_files_for_dup,
        mark_files_absent: Callable[[object, Sequence[int]], None] = _mark_files_absent,
        scanner_factory: Callable[[DuplicateScanConfig], DuplicateScanner] | None = None,
        generate_thumbnail: Callable[..., None] = _generate_thumbnail,
        get_thumbnail: Callable[[Path, int, int], object] = _get_thumbnail,
        cache_dir_factory: Callable[[], Path] = _get_cache_dir,
    ) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path) if db_path is not None else Path(_get_db_path())
        self._connection_factory = connection_factory
        self._iter_files_for_dup = iter_files_for_dup
        self._mark_files_absent = mark_files_absent
        self._scanner_factory = scanner_factory or (lambda config: DuplicateScanner(config))
        self._generate_thumbnail = generate_thumbnail
        self._get_thumbnail = get_thumbnail
        self._cache_dir_factory = cache_dir_factory

    @property
    def db_path(self) -> Path:
        """Return the duplicate detection database path."""

        return self._db_path

    def open_connection(self, db_path: Path | None = None):
        """Return a new database connection."""

        target = Path(db_path) if db_path is not None else self._db_path
        return self._connection_factory(target)

    def iter_files_for_dup(self, connection: object, path_like: str | None):
        """Yield file metadata for duplicate scanning."""

        return list(self._iter_files_for_dup(connection, path_like))

    def mark_files_absent(self, connection: object, file_ids: Sequence[int]) -> None:
        """Mark files as absent in the repository."""

        self._mark_files_absent(connection, list(file_ids))

    def build_clusters(self, config: DuplicateScanConfig, files: Iterable[Sequence[object]]):
        """Create duplicate clusters using the configured scanner factory."""

        scanner = self._scanner_factory(config)
        return scanner.build_clusters(files)

    def generate_thumbnail(self, path: Path, cache_dir: Path, *, size: tuple[int, int], format: str = "WEBP") -> None:
        """Generate a cached thumbnail via :func:`utils.image_io.generate_thumbnail`."""

        self._generate_thumbnail(Path(path), Path(cache_dir), size=size, format=format)

    def get_thumbnail(self, path: Path, width: int, height: int):
        """Return an existing thumbnail pixmap or image handle."""

        return self._get_thumbnail(Path(path), width, height)

    def thumbnail_cache_dir(self) -> Path:
        """Return the directory used for cached thumbnails."""

        return self._cache_dir_factory() / "thumbs"


__all__ = ["DupViewModel"]
