"""Item delegates used by tag search result views."""

from __future__ import annotations

import html
from collections.abc import Callable, Iterable

from PyQt6.QtCore import QEvent, QModelIndex, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QMouseEvent, QPalette, QPixmap, QTextDocument, QTextOption
from PyQt6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QTableView, QWidget

from ui.tag_rendering import (
    _SCORE_COLOR,
    _TAG_LIST_ROLE,
    TagDisplayEntry,
    coerce_category,
    pick_highlight_colors,
    render_tag_html,
)


class HoverRowTableView(QTableView):
    """QTableView that tracks the hovered row for row-level hover highlighting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hover_row: int = -1
        self.viewport().setMouseTracking(True)

    @property
    def hover_row(self) -> int:
        return self._hover_row

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        index = self.indexAt(event.pos())
        new_row = index.row() if index.isValid() else -1
        if new_row != self._hover_row:
            self._hover_row = new_row
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if self._hover_row != -1:
            self._hover_row = -1
            self.viewport().update()
        super().leaveEvent(event)


def _apply_row_hover(opt: QStyleOptionViewItem, index: QModelIndex, hover_row: int) -> None:
    if hover_row >= 0 and index.row() == hover_row:
        opt.state = opt.state | QStyle.StateFlag.State_MouseOver
    else:
        opt.state = opt.state & ~QStyle.StateFlag.State_MouseOver


class HoverAwareDelegate(QStyledItemDelegate):
    """Default delegate that applies row-level hover highlighting."""

    def paint(self, painter: object, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if isinstance(opt.widget, HoverRowTableView):
            _apply_row_hover(opt, index, opt.widget.hover_row)
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)


def grid_caption_lines(text: str) -> list[str]:
    """Return at most two caption lines for a grid thumbnail."""

    raw_lines = str(text or "").splitlines() or [""]
    if len(raw_lines) >= 2:
        return [raw_lines[0], " ".join(line for line in raw_lines[1:] if line)]
    return [raw_lines[0]]


def should_paint_text_background(red: int, green: int, blue: int, *, threshold: float = 128.0) -> bool:
    """Return whether text should receive a backing fill over a dark base."""

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return luminance < threshold


class WrappingItemDelegate(QStyledItemDelegate):
    """Render text on multiple lines instead of eliding."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        wrap_mode: QTextOption.WrapMode = QTextOption.WrapMode.WrapAnywhere,
    ) -> None:
        super().__init__(parent)
        self._wrap_mode = wrap_mode

    def _create_document(self, option: QStyleOptionViewItem, text: str) -> QTextDocument:
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(option.font)
        text_option = QTextOption()
        text_option.setWrapMode(self._wrap_mode)
        alignment = option.displayAlignment or Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        text_option.setAlignment(alignment)
        doc.setDefaultTextOption(text_option)
        safe = html.escape(text)
        doc.setHtml(f'<span style="color:{_SCORE_COLOR};">{safe}</span>')
        return doc

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if isinstance(opt.widget, HoverRowTableView):
            _apply_row_hover(opt, index, opt.widget.hover_row)
        text = opt.text
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        if not text:
            return

        doc = self._create_document(opt, text)
        rect = opt.rect
        painter.save()
        painter.translate(rect.topLeft())
        doc.setTextWidth(rect.width())
        doc.drawContents(painter, QRectF(0, 0, rect.width(), rect.height()))
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        if not text:
            return super().sizeHint(option, index)
        doc = self._create_document(opt, text)
        available_width = option.rect.width()
        if available_width <= 0 and option.widget is not None:
            available_width = option.widget.width()
        if available_width > 0:
            doc.setTextWidth(available_width)
        size = doc.size().toSize()
        size.setHeight(size.height() + 4)
        return size


class HighlightDelegate(QStyledItemDelegate):
    """Render text with highlighted substrings supplied by a provider."""

    def __init__(
        self,
        terms_provider: Callable[[], Iterable[str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._terms_provider = terms_provider

    @staticmethod
    def _to_html_with_highlight(
        text: str,
        terms: list[str],
        tags: Iterable[TagDisplayEntry] | None,
        *,
        bg: str,
        fg: str,
    ) -> str:
        term_set = {term.lower() for term in terms if term}
        if tags:
            parts: list[str] = []
            for entry in tags:
                if not entry:
                    continue
                if isinstance(entry, (list, tuple)):
                    name = str(entry[0])
                    score_value = entry[1] if len(entry) > 1 else None
                    category_value = entry[2] if len(entry) > 2 else None
                else:
                    continue
                try:
                    score = float(score_value)
                except (TypeError, ValueError):
                    continue
                highlighted = term_set and name.lower() in term_set
                parts.append(
                    render_tag_html(
                        name,
                        score,
                        coerce_category(category_value),
                        highlight=bool(highlighted),
                        highlight_bg=bg,
                        highlight_fg=fg,
                    )
                )
            if parts:
                return ", ".join(parts)
        return html.escape(text or "")

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if isinstance(opt.widget, HoverRowTableView):
            _apply_row_hover(opt, index, opt.widget.hover_row)
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        text = str(index.data() or "")
        raw_tags = index.data(int(_TAG_LIST_ROLE))
        tags = list(raw_tags) if raw_tags else []
        terms = list(self._terms_provider() or [])
        background, foreground = pick_highlight_colors(option.palette)
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(opt.font)
        doc.setHtml(self._to_html_with_highlight(text, terms, tags, bg=background, fg=foreground))
        rect = option.rect
        painter.save()
        painter.translate(rect.topLeft())
        doc.setTextWidth(rect.width())
        doc.drawContents(painter, QRectF(0, 0, rect.width(), rect.height()))
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        text = str(index.data() or "")
        raw_tags = index.data(int(_TAG_LIST_ROLE))
        tags = list(raw_tags) if raw_tags else []
        terms = list(self._terms_provider() or [])
        background, foreground = pick_highlight_colors(option.palette)
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(option.font)
        doc.setHtml(self._to_html_with_highlight(text, terms, tags, bg=background, fg=foreground))
        available_width = option.rect.width()
        if available_width <= 0 and option.widget is not None:
            available_width = option.widget.width()
        if available_width > 0:
            doc.setTextWidth(available_width)
        size = doc.size().toSize()
        size.setHeight(size.height() + 4)
        return size


class GridThumbDelegate(QStyledItemDelegate):
    """Render thumbnails with captions in the grid view."""

    def __init__(self, thumb_size: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thumb = thumb_size

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        fm = option.fontMetrics
        text_height = fm.lineSpacing() * 2 + 10
        return QSize(self._thumb + 48, self._thumb + text_height)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        style = opt.widget.style() if opt.widget else QApplication.style()
        painter.save()
        style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelItemViewItem, opt, painter, opt.widget)
        painter.restore()

        rect = opt.rect
        pix = index.data(Qt.ItemDataRole.DecorationRole)
        icon_bottom = rect.y() + self._thumb
        if isinstance(pix, QPixmap) and not pix.isNull():
            thumb = pix.scaled(
                self._thumb,
                self._thumb,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = rect.x() + (rect.width() - thumb.width()) // 2
            y = rect.y()
            painter.drawPixmap(x, y, thumb)
            icon_bottom = y + thumb.height()

        available_height = max(0, rect.y() + rect.height() - icon_bottom - 4)
        text_rect = QRect(rect.x() + 6, icon_bottom + 2, max(0, rect.width() - 12), available_height)
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            return

        palette = opt.palette
        color_group = palette.currentColorGroup()
        is_selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        text_role = QPalette.ColorRole.HighlightedText if is_selected else QPalette.ColorRole.Text
        text_color = palette.color(color_group, text_role)

        base_role = QPalette.ColorRole.Highlight if is_selected else QPalette.ColorRole.Base
        base_color = palette.color(color_group, base_role)
        if should_paint_text_background(base_color.red(), base_color.green(), base_color.blue()):
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(base_color)
            painter.drawRoundedRect(text_rect, 4, 4)
            painter.restore()

        fm = opt.fontMetrics
        lines = grid_caption_lines(str(index.data(Qt.ItemDataRole.DisplayRole) or ""))
        lines = [fm.elidedText(line, Qt.TextElideMode.ElideRight, text_rect.width()) for line in lines]
        lines = [line for line in lines if line] or [""]

        total_height = fm.lineSpacing() * len(lines)
        y_start = text_rect.y() + max(0, text_rect.height() - total_height)

        painter.save()
        painter.setPen(text_color)
        for i, line in enumerate(lines):
            line_rect = QRect(text_rect.x(), y_start + i * fm.lineSpacing(), text_rect.width(), fm.lineSpacing())
            painter.drawText(line_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, line)
        painter.restore()


__all__ = [
    "GridThumbDelegate",
    "HighlightDelegate",
    "HoverAwareDelegate",
    "HoverRowTableView",
    "WrappingItemDelegate",
    "grid_caption_lines",
    "should_paint_text_background",
]
