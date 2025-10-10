"""View model exposing tag-search related operations for the UI layer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

from PyQt6.QtCore import QObject

from core.config import load_settings as _load_settings
from core.pipeline import IndexProgress, retag_all as _retag_all, retag_query as _retag_query, run_index_once as _run_index_once, scan_and_tag as _scan_and_tag
from core.query import extract_positive_tag_terms as _extract_positive_tag_terms, translate_query as _translate_query
from core.settings import PipelineSettings
from db.connection import get_conn as _get_conn
from db.repository import _load_tag_thresholds as _load_tag_thresholds, iter_paths_for_search as _iter_paths_for_search, list_tag_names as _list_tag_names, search_files as _search_files
from utils.paths import ensure_dirs as _ensure_dirs, get_db_path as _get_db_path
from utils.search_export import make_export_dir as _make_export_dir


class TagsViewModel(QObject):
    """Provide backend interactions for :class:`ui.tags_tab.TagsTab`."""

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        db_path: Path | None = None,
        connection_factory: Callable[[Path], object] = _get_conn,
        run_index_once: Callable[..., dict[str, object]] = _run_index_once,
        scan_and_tag: Callable[..., dict[str, object]] = _scan_and_tag,
        retag_query: Callable[[Path, str, Sequence[object]], int] = _retag_query,
        retag_all: Callable[[Path, bool, PipelineSettings], int] = _retag_all,
        load_settings: Callable[[], PipelineSettings] = _load_settings,
        load_thresholds: Callable[[object], dict[str, float]] = _load_tag_thresholds,
        list_tag_names: Callable[[object], Iterable[str]] = _list_tag_names,
        search_files: Callable[..., Iterable] = _search_files,
        iter_paths_for_search: Callable[[object, str], Iterable[str]] = _iter_paths_for_search,
        ensure_directories: Callable[[], None] = _ensure_dirs,
        make_export_dir: Callable[[str], Path] = _make_export_dir,
        translate_query: Callable[[str], str] = _translate_query,
        extract_positive_terms: Callable[[str], set[str]] = _extract_positive_tag_terms,
    ) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path) if db_path is not None else Path(_get_db_path())
        self._connection_factory = connection_factory
        self._run_index_once = run_index_once
        self._scan_and_tag = scan_and_tag
        self._retag_query = retag_query
        self._retag_all = retag_all
        self._load_settings = load_settings
        self._load_thresholds = load_thresholds
        self._list_tag_names = list_tag_names
        self._search_files = search_files
        self._iter_paths_for_search = iter_paths_for_search
        self._ensure_directories = ensure_directories
        self._make_export_dir = make_export_dir
        self._translate_query = translate_query
        self._extract_positive_terms = extract_positive_terms

    @property
    def db_path(self) -> Path:
        """Return the active SQLite database path."""

        return self._db_path

    def open_connection(self, db_path: Path | None = None):
        """Return a new database connection using the configured factory."""

        target = Path(db_path) if db_path is not None else self._db_path
        return self._connection_factory(target)

    def ensure_directories(self) -> None:
        """Ensure required filesystem directories exist."""

        self._ensure_directories()

    def make_export_dir(self, query: str) -> Path:
        """Return a directory suitable for exporting search results."""

        return self._make_export_dir(query)

    def load_settings(self) -> PipelineSettings:
        """Load the persisted pipeline settings."""

        return self._load_settings()

    def load_tag_thresholds(self, connection: object) -> dict[str, float]:
        """Return configured tag thresholds from the database."""

        return self._load_thresholds(connection)

    def list_tag_names(self, connection: object) -> list[str]:
        """Return known tag names from the database."""

        return list(self._list_tag_names(connection))

    def search_files(
        self,
        connection: object,
        where_sql: str,
        params: Sequence[object] | list[object] | tuple[object, ...],
        **kwargs,
    ) -> list:
        """Execute a search query via the repository helper."""

        return list(self._search_files(connection, where_sql, params, **kwargs))

    def iter_paths_for_search(self, connection: object, query: str) -> list[str]:
        """Yield filesystem paths for the given search query."""

        return list(self._iter_paths_for_search(connection, query))

    def run_index_once(
        self,
        db_path: Path,
        *,
        settings: PipelineSettings | None = None,
        progress_cb: Callable[[IndexProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        """Run the indexing pipeline once using the configured callable."""

        return self._run_index_once(
            Path(db_path),
            settings=settings,
            progress_cb=progress_cb,
            is_cancelled=is_cancelled,
        )

    def scan_and_tag(
        self,
        folder: Path,
        *,
        recursive: bool = True,
        batch_size: int = 8,
        hard_delete_missing: bool = False,
    ) -> dict[str, object]:
        """Execute :func:`core.pipeline.scan_and_tag`."""

        return dict(
            self._scan_and_tag(
                Path(folder),
                recursive=recursive,
                batch_size=batch_size,
                hard_delete_missing=hard_delete_missing,
            )
        )

    def retag_query(self, db_path: Path, predicate: str, params: Sequence[object]) -> int:
        """Retag files matching the given predicate."""

        return self._retag_query(Path(db_path), predicate, list(params))

    def retag_all(self, db_path: Path, *, force: bool, settings: PipelineSettings) -> int:
        """Retag the entire library using the current pipeline settings."""

        return self._retag_all(Path(db_path), force, settings)

    def translate_query(self, query: str, **kwargs) -> object:
        """Translate user search text into SQL fragments."""

        return self._translate_query(query, **kwargs)

    def extract_positive_terms(self, query: str) -> set[str]:
        """Extract tags that must be present from the parsed query."""

        return self._extract_positive_terms(query)


__all__ = ["TagsViewModel"]
