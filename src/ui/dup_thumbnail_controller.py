"""Thumbnail queue and binding controller for duplicate results."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThreadPool
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from dup.scanner import DuplicateClusterEntry
from ui.dup_tree_state import should_start_thumbnail, unique_pending_thumbnail_keys
from ui.dup_widgets import ThumbPanel, ThumbTile
from ui.dup_workers import ThumbJob, ThumbSignals
from ui.viewmodels import DupViewModel


class DupThumbnailController:
    """Manage duplicate thumbnail requests, worker queue, and tile bindings."""

    def __init__(
        self,
        *,
        owner: QWidget,
        tree: QTreeWidget,
        view_model: DupViewModel,
        icon_size: QSize,
        placeholder_icon: QIcon,
        budget: int,
        logger: logging.Logger,
        debug_enabled: Callable[[], bool],
    ) -> None:
        self._tree = tree
        self._view_model = view_model
        self._icon_size = icon_size
        self._placeholder_icon = placeholder_icon
        self._budget = budget
        self._log = logger
        self._debug_enabled = debug_enabled
        self.bindings: dict[str, list[ThumbTile]] = {}
        self.inflight: set[str] = set()
        self.pending: deque[str] = deque()
        self.done: set[str] = set()
        self.signals = ThumbSignals()
        self.pool = QThreadPool(owner)
        self.pool.setMaxThreadCount(2)

    def set_icon_size(self, icon_size: QSize) -> None:
        """Update the thumbnail size used for future requests."""

        self._icon_size = icon_size

    def reset(self) -> None:
        """Clear queue state and widget bindings."""

        self.bindings.clear()
        self.inflight.clear()
        self.pending.clear()
        self.done.clear()

    def bind_tile(self, tile: ThumbTile, path: Path) -> None:
        """Bind a tile to a thumbnail path and apply cached data if available."""

        key = str(path)
        self.bindings.setdefault(key, []).append(tile)
        if key in self.done:
            try:
                pix = self._view_model.get_thumbnail(Path(key), self._icon_size.width(), self._icon_size.height())
                tile.set_pixmap(pix, self._placeholder_icon)
            except Exception:
                tile.set_pixmap(None, self._placeholder_icon)

    def queue(self, path: Path) -> None:
        """Queue a path for thumbnail loading if it is not already active."""

        key = str(path)
        if key in self.inflight or key in self.done:
            return
        if key not in self.pending:
            self.pending.append(key)

    def maybe_start_more(self) -> None:
        """Start queued thumbnail jobs up to the active pool capacity."""

        slots = self.pool.maxThreadCount() - self.pool.activeThreadCount()
        slots = min(slots, self._budget)
        cache_dir = self._view_model.thumbnail_cache_dir()
        size = (self._icon_size.width(), self._icon_size.height())

        while slots > 0 and self.pending:
            key = self.pending.popleft()
            if not should_start_thumbnail(key, inflight=self.inflight, done=self.done):
                continue
            path = Path(key)
            job = ThumbJob(self._view_model, path, size, cache_dir, self.signals)
            self.inflight.add(key)
            self.pool.start(job)
            slots -= 1

    def request_visible(
        self,
        *,
        ensure_panel: Callable[[QTreeWidgetItem, bool], None],
        panel_of_group: Callable[[QTreeWidgetItem], ThumbPanel | None],
    ) -> None:
        """Queue thumbnails for tiles visible in the tree viewport."""

        viewport = self._tree.viewport()
        rect = viewport.rect()
        wanted: list[str] = []

        for index in range(self._tree.topLevelItemCount()):
            group = self._tree.topLevelItem(index)
            if not group.isExpanded():
                continue
            ensure_panel(group, False)
            panel = panel_of_group(group)
            if panel is None:
                continue
            for tile in panel.visible_tiles_in(viewport, rect):
                key = str(tile.path)
                if key not in self.inflight and key not in self.done:
                    wanted.append(key)

        self.pending.clear()
        self.pending.extend(unique_pending_thumbnail_keys(wanted, inflight=self.inflight, done=self.done))
        self.maybe_start_more()

    def schedule_visible_items(self, visible_items: list[QTreeWidgetItem]) -> None:
        """Queue thumbnails for legacy tree item rows."""

        for item in visible_items:
            entry = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(entry, DuplicateClusterEntry):
                self.queue(entry.file.path)

    def prune_collapsed_group(self, item: QTreeWidgetItem) -> None:
        """Remove stale bindings for a group that is being collapsed."""

        to_unbind: list[str] = []
        for index in range(item.childCount()):
            child = item.child(index)
            entry = child.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(entry, DuplicateClusterEntry):
                to_unbind.append(str(entry.file.path))
        for key in to_unbind:
            items = self.bindings.get(key)
            if not items:
                continue
            self.bindings[key] = [tile for tile in items if tile is not None and tile.parent() is not None]
            if not self.bindings[key]:
                self.bindings.pop(key, None)

    def apply_done(self, path_str: str, qimg) -> None:
        """Apply a completed thumbnail result to all bound tiles."""

        self.inflight.discard(path_str)
        targets = self.bindings.get(path_str)
        if not targets:
            self.done.add(path_str)
            self.maybe_start_more()
            return

        try:
            if qimg is not None:
                pix = QPixmap.fromImage(qimg)
            else:
                pix = self._view_model.get_thumbnail(
                    Path(path_str),
                    self._icon_size.width(),
                    self._icon_size.height(),
                )
        except Exception:
            pix = None

        for tile in targets:
            tile.set_pixmap(pix, self._placeholder_icon)

        self.done.add(path_str)
        self.maybe_start_more()
        if self._debug_enabled():
            self._log.info("thumb applied: %s", path_str)


__all__ = ["DupThumbnailController"]
