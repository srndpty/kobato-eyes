"""Reusable widgets for duplicate cluster displays."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QCheckBox, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from dup.scanner import DuplicateClusterEntry


def format_duplicate_size(value: int | None) -> str:
    """Format a byte size for duplicate result metadata."""

    if value is None or value <= 0:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.1f} {units[index]}"


def format_duplicate_resolution(width: int | None, height: int | None) -> str:
    """Format image dimensions for duplicate result metadata."""

    if not width or not height:
        return "-"
    return f"{width}×{height}"


class ThumbTile(QFrame):
    """Tile widget for one duplicate candidate."""

    toggled = pyqtSignal(bool)

    def __init__(self, entry: DuplicateClusterEntry, icon_size: QSize, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ThumbTile")
        self.entry = entry
        self.icon_size = icon_size
        self.path = entry.file.path
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            """
            #ThumbTile { background: transparent; }
            #ThumbTile:hover { background: rgba(255,255,255,0.04); border-radius: 6px; }
            """
        )

        self.thumb = QLabel(self)
        self.thumb.setFixedSize(icon_size)
        self.thumb.setScaledContents(False)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.check = QCheckBox(self)
        keeper_id = parent.property("keeper_id") if parent is not None else None
        state = Qt.CheckState.Unchecked if entry.file.file_id == keeper_id else Qt.CheckState.Checked
        self.check.setCheckState(state)

        self.meta1 = QLabel(self)
        self.meta2 = QLabel(self)
        for label in (self.meta1, self.meta2):
            label.setWordWrap(False)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.thumb, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.meta1)
        layout.addWidget(self.meta2)

        size_text = format_duplicate_size(entry.file.size)
        resolution_text = format_duplicate_resolution(entry.file.width, entry.file.height)
        hamming_text = "-" if entry.best_hamming is None else f"H:{entry.best_hamming}"
        self.meta1.setText(f"{size_text}   {resolution_text}   {hamming_text}")

        metrics = QFontMetrics(self.font())
        folder = str(entry.file.path.parent)
        self.meta2.setText(metrics.elidedText(folder, Qt.TextElideMode.ElideMiddle, icon_size.width()))

        self.check.raise_()
        self.check.move(6, 6)
        self.check.stateChanged.connect(lambda _state: self.toggled.emit(self.is_checked()))

    def set_pixmap(self, pixmap: QPixmap | None, placeholder: QIcon) -> None:
        """Apply a pixmap while preserving aspect ratio."""

        target = self.icon_size
        if pixmap is None or pixmap.isNull():
            self.thumb.setPixmap(placeholder.pixmap(target))
            return

        scaled = pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        canvas = QPixmap(target)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        x = (target.width() - scaled.width()) // 2
        y = (target.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        self.thumb.setPixmap(canvas)

    def is_checked(self) -> bool:
        """Return whether this tile is selected for deletion."""

        return self.check.checkState() == Qt.CheckState.Checked

    def set_checked(self, checked: bool) -> None:
        """Set the checked state."""

        self.check.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)


class ThumbPanel(QWidget):
    """Manual grid layout for duplicate thumbnail tiles."""

    sizeHintChanged = pyqtSignal()

    def __init__(
        self,
        entries: list[DuplicateClusterEntry],
        keeper_id: int,
        icon_size: QSize,
        col_gap: int = 10,
        row_gap: int = 12,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_size = icon_size
        self._col_gap = col_gap
        self._row_gap = row_gap
        self.tiles: list[ThumbTile] = []
        self.setProperty("keeper_id", keeper_id)
        for entry in entries:
            tile = ThumbTile(entry, icon_size, parent=self)
            self.tiles.append(tile)
            tile.setParent(self)
            tile.show()
        self._cols = 1
        self._tile_size = QSize(icon_size.width() + 8, icon_size.height() + 8 + 2 * self.fontMetrics().height() + 12)
        self._content_h = self._tile_size.height() + 8
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self._tile_size.height() + 8)

    def _compute_cols(self) -> int:
        width = max(1, self.width())
        cell_width = self._tile_size.width() + self._col_gap
        return max(1, width // cell_width)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        cols = self._compute_cols()
        if cols != self._cols:
            self._cols = cols
        cell_width = self._tile_size.width() + self._col_gap
        cell_height = self._tile_size.height() + self._row_gap
        for index, tile in enumerate(self.tiles):
            row = index // self._cols
            column = index % self._cols
            tile.setGeometry(column * cell_width, row * cell_height, self._tile_size.width(), self._tile_size.height())

        rows = (len(self.tiles) + self._cols - 1) // self._cols
        self._content_h = rows * cell_height
        self.setMinimumHeight(rows * cell_height)
        self.updateGeometry()
        self.sizeHintChanged.emit()

    def sizeHint(self) -> QSize:
        return QSize(self._tile_size.width(), self._content_h)

    def visible_tiles_in(self, viewport: QWidget, tree_viewport_rect: QRect) -> list[ThumbTile]:
        """Return tiles intersecting the tree viewport rectangle."""

        visible: list[ThumbTile] = []
        panel_top_left = self.mapTo(viewport, QPoint(0, 0))
        for tile in self.tiles:
            rect = tile.geometry().translated(panel_top_left)
            if rect.intersects(tree_viewport_rect):
                visible.append(tile)
        return visible


__all__ = ["ThumbPanel", "ThumbTile", "format_duplicate_resolution", "format_duplicate_size"]
