"""Tree binding helpers for duplicate result groups."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from PyQt6.QtCore import QPoint, QSize, Qt, QTimer
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

from dup.scanner import DuplicateCluster, DuplicateClusterEntry
from ui.dup_cluster_update import sort_entries_for_display
from ui.dup_tree_state import default_checked_entries
from ui.dup_widgets import ThumbPanel, ThumbTile


class DupTreeController:
    """Coordinate duplicate tree item and thumbnail-panel bindings."""

    def __init__(self, tree: QTreeWidget, icon_size: QSize) -> None:
        self._tree = tree
        self._icon_size = icon_size

    def set_icon_size(self, icon_size: QSize) -> None:
        """Update the panel icon size for future panel construction."""

        self._icon_size = icon_size

    def panel_of_group(self, group_item: QTreeWidgetItem) -> ThumbPanel | None:
        """Return the thumbnail panel attached to ``group_item`` if present."""

        if group_item.childCount() == 0:
            return None
        child = group_item.child(0)
        widget = self._tree.itemWidget(child, 0)
        return widget if isinstance(widget, ThumbPanel) else None

    def visible_items(self) -> list[QTreeWidgetItem]:
        """Return visible tree items sampled down the viewport."""

        viewport = self._tree.viewport()
        height = viewport.height()
        y = 0
        items: list[QTreeWidgetItem] = []
        seen: set[QTreeWidgetItem] = set()
        while y < height:
            index = self._tree.indexAt(QPoint(10, y))
            if not index.isValid():
                break
            item = self._tree.itemFromIndex(index)
            if item and item not in seen:
                items.append(item)
                seen.add(item)
            rect = self._tree.visualItemRect(item)
            step = rect.height() if rect.height() > 0 else (self._icon_size.height() + 12)
            y += step
        return items

    def iter_checked_entries(self) -> Iterator[DuplicateClusterEntry]:
        """Yield entries currently marked for deletion."""

        for index in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(index)
            if top is None:
                continue
            cluster = top.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(cluster, DuplicateCluster):
                continue

            panel = self.panel_of_group(top)
            if panel is not None:
                for tile in panel.tiles:
                    if tile.is_checked():
                        yield tile.entry
                continue

            for entry in default_checked_entries(cluster):
                yield entry

    def build_children_for_cluster(
        self,
        parent_item: QTreeWidgetItem,
        cluster: DuplicateCluster,
        *,
        bind_tile: Callable[[ThumbTile, Path], None],
        update_actions: Callable[[], None],
    ) -> None:
        """Build and bind the thumbnail panel for one duplicate cluster."""

        entries = sort_entries_for_display(cluster.files, cluster.keeper_id)
        panel_item = QTreeWidgetItem(parent_item)
        panel_item.setFirstColumnSpanned(True)
        panel = ThumbPanel(entries, cluster.keeper_id, self._icon_size, parent=self._tree)
        self._tree.setItemWidget(panel_item, 0, panel)

        def _sync_size_hint() -> None:
            panel_item.setSizeHint(0, panel.sizeHint())
            self._tree.doItemsLayout()
            self._tree.viewport().update()

        panel.sizeHintChanged.connect(_sync_size_hint)
        QTimer.singleShot(0, _sync_size_hint)

        for tile in panel.tiles:
            bind_tile(tile, tile.path)
            tile.toggled.connect(lambda _=None: update_actions())
        update_actions()


__all__ = ["DupTreeController"]
