"""Background helpers shared by the tag search tab."""

from __future__ import annotations

import itertools
import shutil
import sys
from contextlib import closing
from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import QObject, QRunnable, Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

from ui.file_actions import trash_path
from ui.viewmodels import TagsViewModel


def _trash_path(path: Path) -> None:
    """Trash *path*, honoring the legacy ``ui.tags_tab.trash_path`` patch point."""

    tags_tab = sys.modules.get("ui.tags_tab")
    action = getattr(tags_tab, "trash_path", trash_path) if tags_tab is not None else trash_path
    action(path)


class _ElidingLabel(QLabel):
    """幅に収まらないテキストを...で中間省略し、フルテキストはツールチップに出す。"""

    def __init__(self, text: str = "", *, mode=Qt.TextElideMode.ElideMiddle, parent: QWidget | None = None):
        super().__init__(text, parent)
        self._full = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(480)  # ダイアログ幅の目安（お好みで）
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def set_full_text(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._apply_elide()

    def resizeEvent(self, ev):  # type: ignore[override]
        super().resizeEvent(ev)
        self._apply_elide()

    def _apply_elide(self) -> None:
        w = max(50, self.width())
        fm = self.fontMetrics()
        elided = fm.elidedText(self._full, self._mode, w)
        super().setText(elided)


# --- ワーカーシグナル ---------------------------------------------------
class _CopySignals(QObject):
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(str, int, int)  # dest_dir, ok_count, ng_count
    error = pyqtSignal(str)


class _DeleteResultSignals(QObject):
    """Signals emitted by a background search-result deletion task."""

    finished = pyqtSignal(list, list, list)


class _DeleteResultRunnable(QRunnable):
    """Move result files to trash and mark their DB rows absent."""

    def __init__(
        self,
        view_model: TagsViewModel,
        db_path: Path,
        *,
        entries: Sequence[tuple[int, Path]],
    ) -> None:
        super().__init__()
        self._view_model = view_model
        self._db_path = Path(db_path)
        self._entries = [(int(file_id), Path(path)) for file_id, path in entries]
        self.signals = _DeleteResultSignals()

    def run(self) -> None:
        """Execute the delete workflow off the GUI thread."""

        trashed: list[tuple[int, str]] = []
        failures: list[tuple[str, int, str, str]] = []
        for file_id, path in self._entries:
            try:
                if not path.is_file():
                    raise FileNotFoundError("Path is not a file")
                _trash_path(path)
                trashed.append((file_id, str(path)))
            except Exception as exc:
                failures.append(("trash", file_id, str(path), str(exc)))

        db_updated_ids: list[int] = []
        if trashed:
            trashed_ids = [file_id for file_id, _ in trashed]
            trashed_paths = {file_id: path for file_id, path in trashed}
            try:
                with closing(self._view_model.open_connection(self._db_path)) as conn:
                    self._view_model.mark_files_absent(conn, trashed_ids)
                db_updated_ids = trashed_ids
            except Exception as exc:
                message = str(exc)
                failures.extend(("db", file_id, trashed_paths[file_id], message) for file_id in trashed_ids)

        self.signals.finished.emit(trashed, db_updated_ids, failures)


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    """Return a non-conflicting destination path inside ``dest_dir``."""

    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    for index in itertools.count(2):
        candidate = dest_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate


# --- バックグラウンドコピー ---------------------------------------------
class _CopyRunnable(QRunnable):
    def __init__(
        self,
        view_model: TagsViewModel,
        db_path: Path,
        query: str,
        dest_dir: Path,
    ) -> None:
        super().__init__()
        self._view_model = view_model
        self.db_path = db_path
        self.query = query
        self.dest_dir = dest_dir
        self.signals = _CopySignals()

    def _unique_dest(self, name: str) -> Path:
        """同名ファイルがある場合は _2, _3... と連番で回避。"""
        return _unique_destination(self.dest_dir, name)

    def run(self) -> None:
        try:
            # 1) パス列挙
            with self._view_model.open_connection(self.db_path) as conn:
                paths = self._view_model.iter_paths_for_search(conn, self.query)
            total = len(paths)
            self.signals.progress.emit(0, total)
            ok = ng = 0

            # 2) コピー
            for idx, p in enumerate(paths, start=1):
                try:
                    src = Path(p)
                    if src.exists():
                        dst = self._unique_dest(src.name)
                        shutil.copy2(src, dst)
                        ok += 1
                    else:
                        ng += 1
                except Exception:
                    ng += 1
                self.signals.progress.emit(idx, total)

            self.signals.finished.emit(str(self.dest_dir), ok, ng)
        except Exception as e:
            self.signals.error.emit(str(e))
