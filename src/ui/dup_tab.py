"""Duplicate detection UI components."""

from __future__ import annotations

import csv
import logging
import os
import platform
import subprocess
import sys
import time
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QtMsgType,
    pyqtSignal,
    qInstallMessageHandler,
)
from PyQt6.QtGui import QAction, QColor, QFontMetrics, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

from core.config import load_settings
from db.connection import get_conn
from db.repository import iter_files_for_dup, mark_files_absent
from dup.scanner import DuplicateCluster, DuplicateClusterEntry, DuplicateFile, DuplicateScanConfig, DuplicateScanner
from ui.dup_refine_parallel import refine_by_pixels_parallel, refine_by_tilehash_parallel
from utils.image_io import generate_thumbnail, get_thumbnail  # 既存ユーティリティを再利用
from utils.paths import get_cache_dir, get_db_path

LOG = logging.getLogger("dup")
if not LOG.handlers:
    LOG.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(threadName)s %(levelname)s: %(message)s"))
    LOG.addHandler(h)
DEBUG_THUMBS = False


def _qt_msg(mode, ctx, msg):
    try:
        level = {
            QtMsgType.QtDebugMsg: "DEBUG",
            QtMsgType.QtInfoMsg: "INFO",
            QtMsgType.QtWarningMsg: "WARNING",
            QtMsgType.QtCriticalMsg: "CRITICAL",
            QtMsgType.QtFatalMsg: "FATAL",
        }[mode]
    except Exception:
        level = "QT"
    LOG.warning(f"QT[{level}] {msg} ({ctx.file}:{ctx.line})")


qInstallMessageHandler(_qt_msg)


class RefinePipelineSignals(QObject):
    progress = pyqtSignal(int, int, str)  # cur, total, stage label
    finished = pyqtSignal(object)  # refined clusters
    canceled = pyqtSignal()
    error = pyqtSignal(str)


class RefinePipelineRunnable(QRunnable):
    def __init__(self, clusters, tile_params, pixel_params):
        super().__init__()
        self.signals = RefinePipelineSignals()
        self._clusters = clusters
        self._tile_params = tile_params or {}
        self._pixel_params = pixel_params or {}
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _is_cancelled(self):
        return self._cancelled

    def run(self):
        try:
            # ---- TileHash (1/2, 2/2) ----
            def tick_tile(done, total, phase):
                stage = "TileHash 1/2" if phase == 1 else "TileHash 2/2"
                self.signals.progress.emit(done, total, stage)

            refined = refine_by_tilehash_parallel(
                self._clusters, tick=tick_tile, is_cancelled=self._is_cancelled, **self._tile_params
            )
            if self._cancelled:
                self.signals.canceled.emit()
                return

            # ---- Pixel MAE ----
            if self._pixel_params:

                def tick_px(done, total):
                    self.signals.progress.emit(done, total, "Pixel MAE")

                refined = refine_by_pixels_parallel(
                    refined, tick=tick_px, is_cancelled=self._is_cancelled, **self._pixel_params
                )
                if self._cancelled:
                    self.signals.canceled.emit()
                    return

            self.signals.finished.emit(refined)
        except Exception as e:
            self.signals.error.emit(str(e))


@dataclass(frozen=True)
class DuplicateScanRequest:
    """Parameters supplied to the duplicate scanning worker."""

    path_like: str | None
    hamming_threshold: int
    size_ratio: float | None
    cosine_threshold: float | None


class ThumbTile(QFrame):
    """256x256 サムネ + 2行テキスト + チェックの小さなタイル."""

    toggled = pyqtSignal(bool)  # 必要なら使う

    def __init__(self, entry: DuplicateClusterEntry, icon_size: QSize, parent=None):
        super().__init__(parent)
        self.setObjectName("ThumbTile")
        self.entry = entry
        self.icon_size = icon_size
        self.path = entry.file.path
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("""
        #ThumbTile { background: transparent; }
        #ThumbTile:hover { background: rgba(255,255,255,0.04); border-radius: 6px; }
        """)

        self.thumb = QLabel(self)
        self.thumb.setFixedSize(icon_size)
        self.thumb.setScaledContents(False)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.check = QCheckBox(self)
        # 既定状態（keeper は Unchecked, それ以外 Checked）
        st = Qt.CheckState.Unchecked if entry.file.file_id == parent.property("keeper_id") else Qt.CheckState.Checked
        self.check.setCheckState(st)

        self.meta1 = QLabel(self)  # 1行目: size,res,ham,cos
        self.meta2 = QLabel(self)  # 2行目: dir (middle elide)
        for lab in (self.meta1, self.meta2):
            lab.setWordWrap(False)
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(self.thumb, alignment=Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self.meta1)
        lay.addWidget(self.meta2)

        # メタ設定
        size_t = DupTab._format_size(entry.file.size)
        res_t = DupTab._format_resolution(entry.file.width, entry.file.height)
        ham_t = "-" if entry.best_hamming is None else f"H:{entry.best_hamming}"
        cos_t = "-" if entry.best_cosine is None else f"C:{entry.best_cosine:.3f}"
        self.meta1.setText(f"{size_t}   {res_t}   {ham_t}   {cos_t}")

        fm = QFontMetrics(self.font())
        folder = str(entry.file.path.parent)
        self.meta2.setText(fm.elidedText(folder, Qt.TextElideMode.ElideMiddle, icon_size.width()))

        # チェックは左上にオーバレイ
        self.check.raise_()
        self.check.move(6, 6)

        self.check.stateChanged.connect(lambda _s: self.toggled.emit(self.is_checked()))

    # サムネ適用
    def set_pixmap(self, pix: QPixmap | None, placeholder: QIcon) -> None:
        target = self.icon_size
        if pix is None or pix.isNull():
            self.thumb.setPixmap(placeholder.pixmap(target))
            return

        # アス比維持でフィット
        scaled = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # レターボックスに中央配置
        canvas = QPixmap(target)
        canvas.fill(Qt.GlobalColor.transparent)  # 透過。灰背景にしたいなら fill(QColor(50,50,50))
        p = QPainter(canvas)
        x = (target.width() - scaled.width()) // 2
        y = (target.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        p.end()
        self.thumb.setPixmap(canvas)

    def is_checked(self) -> bool:
        return self.check.checkState() == Qt.CheckState.Checked

    def set_checked(self, checked: bool) -> None:
        self.check.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)


class ThumbPanel(QWidget):
    """
    グループ内のタイルをグリッドで敷き詰めるパネル。
    QGridLayout でも良いが、パフォーマンスのため手動レイアウトにする。
    """

    sizeHintChanged = pyqtSignal()

    def __init__(
        self,
        entries: list[DuplicateClusterEntry],
        keeper_id: int,
        icon_size: QSize,
        col_gap=10,
        row_gap=12,
        parent=None,
    ):
        super().__init__(parent)
        self._icon_size = icon_size
        self._col_gap = col_gap
        self._row_gap = row_gap
        self.tiles: list[ThumbTile] = []
        self.setProperty("keeper_id", keeper_id)
        for e in entries:
            t = ThumbTile(e, icon_size, parent=self)
            self.tiles.append(t)
            t.setParent(self)
            t.show()
        self._cols = 1
        self._tile_size = QSize(icon_size.width() + 8, icon_size.height() + 8 + 2 * self.fontMetrics().height() + 12)
        self._content_h = self._tile_size.height() + 8  # ← パネル全体の高さを保持
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # ← 重要（縦は sizeHint に従う）
        self.setMinimumHeight(self._tile_size.height() + 8)

    def _compute_cols(self) -> int:
        w = max(1, self.width())
        cell_w = self._tile_size.width() + self._col_gap
        return max(1, w // cell_w)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._relayout()

    def _relayout(self):
        cols = self._compute_cols()
        if cols != self._cols:
            self._cols = cols
        cell_w = self._tile_size.width() + self._col_gap
        cell_h = self._tile_size.height() + self._row_gap
        for i, t in enumerate(self.tiles):
            r = i // self._cols
            c = i % self._cols
            x = c * cell_w
            y = r * cell_h
            t.setGeometry(x, y, self._tile_size.width(), self._tile_size.height())

        rows = (len(self.tiles) + self._cols - 1) // self._cols
        self._content_h = rows * cell_h
        self.setMinimumHeight(rows * cell_h)
        self.updateGeometry()
        self.sizeHintChanged.emit()

    def sizeHint(self):
        # 幅はツリー側では使われないのでダミーでOK。高さだけ正確に返す
        return QSize(self._tile_size.width(), self._content_h)
        # self._relayout()
        # return super().sizeHint()

    def visible_tiles_in(self, viewport: QWidget, tree_viewport_rect: QRect) -> list[ThumbTile]:
        """ツリーの viewport 座標系で見えているタイルを返す。"""
        vis: list[ThumbTile] = []
        # パネルの (0,0) をツリー viewport 座標へ
        panel_top_left = self.mapTo(viewport, QPoint(0, 0))
        for t in self.tiles:
            r = t.geometry().translated(panel_top_left)
            if r.intersects(tree_viewport_rect):
                vis.append(t)
        return vis


class DuplicateScanSignals(QObject):
    """Signals emitted by the scanning worker."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class DuplicateScanRunnable(QRunnable):
    """Background runnable that loads file metadata and clusters duplicates."""

    def __init__(self, db_path: Path, request: DuplicateScanRequest) -> None:
        super().__init__()
        self._db_path = db_path
        self._request = request
        self.signals = DuplicateScanSignals()

    def run(self) -> None:
        try:
            LOG.info("scan: start")
            settings = load_settings()
            model_name = settings.model_name

            # conn はこの with（closing）ブロック内でだけ使う
            with closing(get_conn(self._db_path)) as conn:
                rows = list(iter_files_for_dup(conn, self._request.path_like, model_name=model_name))
                LOG.info("scan: rows loaded = %d", len(rows))

            total = len(rows)
            self.signals.progress.emit(0, total)

            files: list[DuplicateFile] = []
            for index, row in enumerate(rows, start=1):
                try:
                    files.append(DuplicateFile.from_row(row))
                except ValueError:
                    continue
                if index % 500 == 0:
                    self.signals.progress.emit(index, total)
            self.signals.progress.emit(total, total)

            config = DuplicateScanConfig(
                hamming_threshold=self._request.hamming_threshold,
                size_ratio=self._request.size_ratio,
                cosine_threshold=self._request.cosine_threshold,
            )

            LOG.info("cluster: building ...")
            t0 = time.perf_counter()
            clusters = DuplicateScanner(config).build_clusters(files)
            LOG.info("cluster: done; n=%d, %.2fs", len(clusters), time.perf_counter() - t0)

            self.signals.finished.emit(clusters)

        except Exception as exc:
            LOG.exception("scan worker crashed: %s", exc)
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                pass


class _ThumbSignals(QObject):
    done = pyqtSignal(str, object)  # path(str) 単位で完了通知


class _ThumbJob(QRunnable):
    """
    1ファイル分のサムネをディスクキャッシュに生成するだけ（Pillow使用）。
    QPixmapはGUIスレッドで作るので、ここでは使わない。
    """

    def __init__(self, path: Path, size: tuple[int, int], cache_dir: Path, signals: _ThumbSignals) -> None:
        super().__init__()
        self._path = path
        self._size = size
        self._cache_dir = cache_dir
        self._signals = signals

    def run(self) -> None:
        qimg = None
        try:
            # 将来の再利用用にディスクサムネは作っておく（失敗してもOK）
            try:
                generate_thumbnail(self._path, self._cache_dir, size=self._size, format="WEBP")
            except Exception as e:
                if DEBUG_THUMBS:
                    print("thumb gen failed:", self._path, e)

            # 直接 QImage を作る（GUIスレッドでの再デコードを避ける）
            with Image.open(self._path) as im:
                im.load()
                im = im.convert("RGB")
                im.thumbnail(self._size, Image.Resampling.LANCZOS)
                qimg = ImageQt(im).copy()  # .copy() でバッファを独立させる
        except Exception as e:
            if DEBUG_THUMBS:
                print("thumb worker error:", self._path, e)
            qimg = None
        finally:
            try:
                self._signals.done.emit(str(self._path), qimg)
                if DEBUG_THUMBS:
                    print("thumb emitted:", self._path, bool(qimg))
            except Exception as e:
                # ここに来ることはほぼ無いが保険
                print("thumb emit failed:", e)


class DupTab(QWidget):
    """Provide controls for duplicate search and clustering."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = get_db_path()
        self._pool = QThreadPool(self)
        self._clusters: list[DuplicateCluster] = []
        self._active_scan: DuplicateScanRunnable | None = None
        self._block_item_changed = False
        self._bulk_populating = False
        self._refine_dialog = None
        self._refine_task = None

        self._hamming_spin = QSpinBox(self)
        self._hamming_spin.setRange(0, 10)
        self._hamming_spin.setValue(0)

        self._ratio_spin = QDoubleSpinBox(self)
        self._ratio_spin.setRange(0.0, 1.0)
        self._ratio_spin.setSingleStep(0.05)
        self._ratio_spin.setDecimals(2)
        self._ratio_spin.setValue(0.90)
        self._ratio_spin.setSpecialValueText("disabled")

        self._cosine_spin = QDoubleSpinBox(self)
        self._cosine_spin.setRange(0.0, 2.0)
        self._cosine_spin.setSingleStep(0.05)
        self._cosine_spin.setDecimals(3)
        self._cosine_spin.setValue(0.20)
        self._cosine_spin.setSpecialValueText("disabled")

        self._scan_button = QPushButton("Scan", self)
        self._mark_button = QPushButton("Mark keep-largest", self)
        self._uncheck_button = QPushButton("Uncheck all", self)
        self._trash_button = QPushButton("Trash checked", self)
        self._export_button = QPushButton("Export CSV", self)

        self._progress = QProgressBar(self)
        self._progress.setFormat("%v / %m")
        self._progress.setMaximum(1)
        self._progress.setValue(0)

        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)
        self.setStyleSheet("""
        #ThumbTile { background: transparent; }   # ← :hover 行を削除
        """)
        self._tree = QTreeWidget(self)
        self._tree.setStyleSheet("""
        QTreeView::item:hover {               /* ← ホバー時の青い塗りを消す */
            background: transparent;
        }
        QTreeView::item:selected:active {     /* ← ついでに選択時の色も弱めたい場合 */
            background: rgba(80, 120, 200, 0.25);
        }
        QTreeView::item:selected:!active {
            background: rgba(80, 120, 200, 0.18);
        }
        QTreeView { outline: 0; }
        """)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # フォーカス枠も出さない
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)

        hdr = self._tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # 追加
        # self._tree.setHeaderLabels(["File", "Size", "Resolution", "Hamming", "Cosine", "Path"])
        self._tree.setAlternatingRowColors(True)
        # ★ アイコンサイズ（Tagsタブに寄せるなら 96x96 や 128x128）
        self._icon_size = QSize(256, 256)
        self._tree.setIconSize(self._icon_size)
        self._tree.setUniformRowHeights(False)

        # サムネの可視範囲リクエストをタイマでデバウンス
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.timeout.connect(self._request_visible_thumbs)
        self._tree.verticalScrollBar().valueChanged.connect(lambda _: self._thumb_timer.start(30))
        self._tree.viewport().installEventFilter(self)

        # サムネ用スレッドプールを控えめに（I/O スパイク防止）
        self._pool.setMaxThreadCount(min(4, os.cpu_count() or 2))

        self._tree.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        sb = self._tree.verticalScrollBar()
        sb.setSingleStep(24)  # 1ホイール刻みのpx。好みで 12〜48 に調整
        self._tree.setAnimated(False)
        self._tree.setSortingEnabled(False)
        self._tree.setExpandsOnDoubleClick(False)

        self._tree.setRootIsDecorated(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu_requested)

        self._grid_spin = QSpinBox(self)
        self._grid_spin.setRange(2, 16)
        self._grid_spin.setValue(8)
        self._tile_spin = QSpinBox(self)
        self._tile_spin.setRange(4, 16)
        self._tile_spin.setValue(8)
        self._maxbits_spin = QSpinBox(self)
        self._maxbits_spin.setRange(0, 128)
        self._maxbits_spin.setValue(8)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Hamming ≤", self))
        controls_layout.addWidget(self._hamming_spin)
        controls_layout.addWidget(QLabel("Size ratio ≥", self))
        controls_layout.addWidget(self._ratio_spin)
        controls_layout.addWidget(QLabel("Cosine ≤", self))
        controls_layout.addWidget(self._cosine_spin)
        controls_layout.addSpacing(16)
        controls_layout.addWidget(QLabel("grid", self))
        controls_layout.addWidget(self._grid_spin)
        controls_layout.addWidget(QLabel("tile", self))
        controls_layout.addWidget(self._tile_spin)
        controls_layout.addWidget(QLabel("max_bits", self))
        controls_layout.addWidget(self._maxbits_spin)
        controls_layout.addStretch(1)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._scan_button)
        button_layout.addWidget(self._mark_button)
        button_layout.addWidget(self._uncheck_button)
        button_layout.addWidget(self._trash_button)
        button_layout.addWidget(self._export_button)
        button_layout.addStretch(1)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._progress)
        status_layout.addWidget(self._status_label, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls_layout)
        layout.addLayout(button_layout)
        layout.addLayout(status_layout)
        layout.addWidget(self._tree, 1)

        self._scan_button.clicked.connect(self._on_scan_clicked)
        self._mark_button.clicked.connect(self._on_mark_keep_largest)
        self._uncheck_button.clicked.connect(self._on_uncheck_all)
        self._trash_button.clicked.connect(self._on_trash_checked)
        self._export_button.clicked.connect(self._on_export_csv)

        # ★ path → [QTreeWidgetItem] の逆引き（サムネ適用に使う）
        self._thumb_bindings: dict[str, list[ThumbTile]] = {}

        # ★ 進行中ジョブの重複投げ防止
        self._thumb_inflight: set[str] = set()

        # ★ サムネワーカー通信用シグナル
        self._thumb_signals = _ThumbSignals()
        self._thumb_signals.done.connect(self._on_thumb_done, Qt.ConnectionType.QueuedConnection)

        # ★ QThreadPool は既存の self._pool を使い回す（QRunnableをstartでOK）
        self._thumb_pool = QThreadPool(self)
        self._thumb_pool.setMaxThreadCount(2)  # サムネは低並列に
        # ★ プレースホルダ（薄グレーの枠）
        self._placeholder_icon = self._make_placeholder_icon(self._icon_size)
        self._tree.itemExpanded.connect(self._on_group_expanded)
        self._tree.itemCollapsed.connect(self._on_group_collapsed)
        self._thumb_pending = deque()  # まだ QThreadPool に start していないキー
        self._thumb_done: set[str] = set()  # アイコン適用済み（or キャッシュ命中）キー
        self._thumb_budget = int(os.environ.get("KE_DUP_THUMB_BUDGET", "12"))
        self._prefetch_rows = int(os.environ.get("KE_DUP_PREFETCH_ROWS", "8"))
        self._update_action_states()

        self._alive = 0
        self._hb = QTimer(self)
        self._hb.setInterval(1000)
        self._hb.timeout.connect(lambda: LOG.debug(f"ui alive {self._alive:=self._alive+1}"))
        self._hb.start()

    def _update_status_summary(self) -> None:
        groups = len(self._clusters)
        files = sum(len(c.files) for c in self._clusters)
        self._status_label.setText(f"{groups} group(s), {files} file(s) detected.")

    def _start_refine_pipeline(self, clusters):
        # パラメータ（好みに応じて UI から取ってもOK）
        tile_params = dict(
            grid=self._grid_spin.value(),
            tile=self._tile_spin.value(),
            max_bits=self._maxbits_spin.value(),
        )
        pixel_params = dict(mae_thr=0.004)

        task = RefinePipelineRunnable(clusters, tile_params, pixel_params)
        self._refine_task = task

        # 進捗ダイアログ
        dlg = QProgressDialog("Refining duplicates…", "Cancel", 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        self._refine_dialog = dlg

        # シグナル接続
        def on_progress(cur, total, stage):
            if total <= 0:
                total = 1
            dlg.setLabelText(f"{stage}  {cur} / {total}")
            if dlg.maximum() != total:
                dlg.setMaximum(total)
            dlg.setValue(cur)

        def on_finished(refined):
            dlg.close()
            self._refine_dialog = None
            self._refine_task = None
            # ここで self._clusters を置き換え → 並べ替え → ツリー構築
            self._clusters = refined
            self._clusters.sort(key=lambda c: (self._cluster_hamming_score(c), len(c.files)), reverse=True)
            self._populate_tree()
            groups = len(self._clusters)
            files = sum(len(c.files) for c in self._clusters)
            self._status_label.setText(f"Refine complete: {groups} group(s), {files} file(s).")
            self._update_action_states()
            self._update_status_summary()

        def on_canceled():
            dlg.close()
            self._refine_dialog = None
            self._refine_task = None
            # キャンセル時は “元の” clusters で表示するか、何もしないか選べます
            self._status_label.setText("Refine canceled.")

        def on_error(msg):
            dlg.close()
            self._refine_dialog = None
            self._refine_task = None
            QMessageBox.warning(self, "Refine failed", msg)
            self._status_label.setText("Refine failed.")

        task.signals.progress.connect(on_progress)
        task.signals.finished.connect(on_finished)
        task.signals.canceled.connect(on_canceled)
        task.signals.error.connect(on_error)

        dlg.canceled.connect(task.cancel)

        dlg.show()
        self._pool.start(task)  # 既存の QThreadPool を使う

    def _auto_expand_visible_groups(self, margin: int = 200) -> None:
        rect = self._tree.viewport().rect().adjusted(0, -margin, 0, +margin)
        budget = int(os.environ.get("KE_DUP_EXPAND_BUDGET", "24"))  # 1ティック上限
        expanded = 0

        idx = self._tree.indexAt(QPoint(10, 10))
        # 画面上から下に向かって可視範囲だけ舐める
        while idx.isValid():
            item = self._tree.itemFromIndex(idx)
            if item is None:
                break
            if item.parent() is None:  # トップレベルのみ
                r = self._tree.visualItemRect(item)
                if r.bottom() < rect.top():
                    idx = self._tree.indexBelow(idx)
                    continue
                if r.top() > rect.bottom():
                    break

                # まずヘッダーを（遅延）装着
                self._ensure_header_built_if_visible(item)

                # 見えたら自動展開
                if not item.isExpanded():
                    item.setExpanded(True)
                    expanded += 1
                    if expanded >= budget:
                        break

                # 展開状態ならパネルも（遅延）構築
                self._ensure_panel_built_if_visible(item, force=False)
            idx = self._tree.indexBelow(idx)

    def _ensure_header_built_if_visible(self, item: QTreeWidgetItem) -> None:
        # すでに付いていれば何もしない
        if self._tree.itemWidget(item, 0) is not None:
            return
        # 画面内±200px のときだけ作る
        vp = self._tree.viewport()
        r = self._tree.visualItemRect(item)
        if not (r.isValid() and r.bottom() >= -200 and r.top() <= (vp.height() + 200)):
            return

        # ここで初めて軽量なヘッダーbarを作る
        bar = QWidget(self._tree)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(8, 2, 8, 2)
        hl.setSpacing(8)
        # タイトルだけ描画にしておく（ラベルは軽い）
        cl: DuplicateCluster = item.data(0, Qt.ItemDataRole.UserRole)
        idx = self._tree.indexOfTopLevelItem(item) + 1
        title = QLabel(f"Group #{idx} ({len(cl.files)} items)", bar)
        btn_keep = QPushButton("Keep largest in group", bar)
        btn_uncheck = QPushButton("Uncheck all in group", bar)
        btn_keep.setFixedHeight(22)
        btn_uncheck.setFixedHeight(22)
        hl.addWidget(title)
        hl.addStretch(1)
        hl.addWidget(btn_keep)
        hl.addWidget(btn_uncheck)
        self._tree.setItemWidget(item, 0, bar)
        item.setFirstColumnSpanned(True)

        # ボタンのハンドラ（クロージャで対象を束縛）
        def _keep_this_group(it=item, cl_=cl):
            self._ensure_panel_built_if_visible(it, force=True)
            panel = self._panel_of_group(it)
            if not panel:
                return
            for t in panel.tiles:
                t.set_checked(t.entry.file.file_id != cl_.keeper_id)
            self._update_action_states()

        def _uncheck_this_group(it=item):
            self._ensure_panel_built_if_visible(it, force=True)
            panel = self._panel_of_group(it)
            if not panel:
                return
            for t in panel.tiles:
                t.set_checked(False)
            self._update_action_states()

        btn_keep.clicked.connect(_keep_this_group)
        btn_uncheck.clicked.connect(_uncheck_this_group)

    def _bind_tile_to_thumb(self, tile: ThumbTile, path: Path) -> None:
        key = str(path)
        self._thumb_bindings.setdefault(key, []).append(tile)
        # 既に done 済みなら即適用
        if key in self._thumb_done:
            try:
                pix = get_thumbnail(key, self._icon_size.width(), self._icon_size.height())
                tile.set_pixmap(pix, self._placeholder_icon)
            except Exception:
                tile.set_pixmap(None, self._placeholder_icon)

    def _maybe_start_more_thumbs(self) -> None:
        slots = self._thumb_pool.maxThreadCount() - self._thumb_pool.activeThreadCount()
        # 1 ティックの上限（スクロール直後のスパイク抑止）
        slots = min(slots, self._thumb_budget)
        cache_dir = get_cache_dir() / "thumbs"
        size = (self._icon_size.width(), self._icon_size.height())

        while slots > 0 and self._thumb_pending:
            key = self._thumb_pending.popleft()
            if key in self._thumb_inflight or key in self._thumb_done:
                continue
            path = Path(key)
            job = _ThumbJob(path, size, cache_dir, self._thumb_signals)
            self._thumb_inflight.add(key)
            self._thumb_pool.start(job)
            slots -= 1

    def _bound_path_of_item(self, item: QTreeWidgetItem) -> Path | None:
        # 可視行は数十件なので O(可視件数) の探索で十分軽い
        for key, lst in self._thumb_bindings.items():
            if item in lst:
                try:
                    return Path(key)
                except Exception:
                    return None
        return None

    def _request_visible_thumbs(self) -> None:
        # ★ 先に可視グループを自動展開
        self._auto_expand_visible_groups(margin=200)

        vp = self._tree.viewport()
        rect = vp.rect()

        # まず可視グループにヘッダーを付ける
        idx = self._tree.indexAt(QPoint(10, 10))
        # インデックスから下方向へ、ビュー外に出るまで
        touched = 0
        while idx.isValid():
            item = self._tree.itemFromIndex(idx)
            if item and item.parent() is None:  # トップレベル
                r = self._tree.visualItemRect(item)
                if r.top() > rect.bottom() + 200:
                    break
                self._ensure_header_built_if_visible(item)  # ★ここ
                # 見えていればパネルも用意（既存関数）
                if item.isExpanded():
                    self._ensure_panel_built_if_visible(item, force=False)
            idx = self._tree.indexBelow(idx)
            touched += 1
            if touched > 2000:  # 万一の無限ループ保険
                break

        # 以降は今までの処理（パネルがあるグループから見えているタイルを集める）
        wanted: list[str] = []
        for i in range(self._tree.topLevelItemCount()):
            g = self._tree.topLevelItem(i)
            if not g.isExpanded():
                continue
            panel = self._panel_of_group(g)
            if not panel:
                continue
            for tile in panel.visible_tiles_in(vp, rect):
                key = str(tile.path)
                if key not in self._thumb_inflight and key not in self._thumb_done:
                    wanted.append(key)

        self._thumb_pending.clear()
        for key in wanted:
            if key not in self._thumb_pending:
                self._thumb_pending.append(key)

        # self._status_label.setText(f"thumb requests (visible): ~{len(self._thumb_pending)}")
        self._maybe_start_more_thumbs()

    def _iter_tiles(self) -> Iterator[ThumbTile]:
        for i in range(self._tree.topLevelItemCount()):
            g = self._tree.topLevelItem(i)
            panel = self._panel_of_group(g)
            if not panel:
                continue
            for t in panel.tiles:
                yield t

    def _cluster_hamming_score(self, cluster: "DuplicateCluster") -> int:
        """
        クラスター内で keeper 以外の best_hamming の最大値を返す。
        値が大きいほど“目視確認の必要性が高い”とみなす。
        すべて None の場合は -1（最下位扱い）を返す。
        """
        vals = []
        for e in cluster.files:
            if e.file.file_id == cluster.keeper_id:
                continue
            if e.best_hamming is not None:
                vals.append(e.best_hamming)
        return max(vals) if vals else -1

    def _make_placeholder_icon(self, size: QSize) -> QIcon:
        img = QPixmap(size)
        img.fill(QColor(50, 50, 50))
        p = QPainter(img)
        p.setPen(QColor(90, 90, 90))
        p.drawRect(1, 1, size.width() - 2, size.height() - 2)
        p.end()
        return QIcon(img)

    def _bind_item_to_thumb(self, item: QTreeWidgetItem, path: Path) -> None:
        key = str(path)
        lst = self._thumb_bindings.setdefault(key, [])
        lst.append(item)

        # ★ 追加：すでに done 済みなら即適用（親で読み終わっているケースを拾う）
        if key in self._thumb_done:
            try:
                pix = get_thumbnail(key, self._icon_size.width(), self._icon_size.height())
                item.setIcon(0, QIcon(pix))
            except Exception:
                item.setIcon(0, self._placeholder_icon)

    def _request_thumb(self, path: Path) -> None:
        key = str(path)
        if key in self._thumb_inflight:
            return
        self._thumb_inflight.add(key)
        cache_dir = get_cache_dir() / "thumbs"
        if DEBUG_THUMBS:
            print(f"thumb enqueue: {key}")  # ← デバッグ
        job = _ThumbJob(path, (self._icon_size.width(), self._icon_size.height()), cache_dir, self._thumb_signals)
        self._thumb_pool.start(job)  # ← こっちのプールで

    # ウィジェット表示時にも走らせる（保険）
    def showEvent(self, ev):
        super().showEvent(ev)
        self._thumb_timer.start(0)

    def eventFilter(self, obj, ev):
        if obj is self._tree.viewport():
            if ev.type() in (
                QEvent.Type.Show,
                QEvent.Type.Resize,
                QEvent.Type.Paint,
                QEvent.Type.UpdateRequest,
            ):
                self._thumb_timer.start(0)

            if (
                ev.type() == QEvent.Type.MouseButtonRelease
                and getattr(ev, "button", None)
                and ev.button() == Qt.MouseButton.LeftButton
            ):
                pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
                item = self._tree.itemAt(pos)
                if item is not None and item.parent() is None:
                    # 親行をクリック。展開矢印(左端 ~18px)は除外
                    idx = self._tree.indexAt(pos)
                    rect = self._tree.visualItemRect(item)
                    clicked_in_arrow = (idx.column() == 0) and ((pos.x() - rect.left()) < 18)
                    if not clicked_in_arrow:
                        item.setExpanded(not item.isExpanded())
                        # 未構築ならここで子を作る（矢印を押さなくても開くので）
                        if (
                            item.childCount() == 1
                            and item.child(0).data(0, Qt.ItemDataRole.UserRole) == "__placeholder__"
                        ):
                            cluster = item.data(0, Qt.ItemDataRole.UserRole)
                            item.takeChildren()
                            self._build_children_for_cluster(item, cluster)
                            self._schedule_visible_thumbs()
        return False

    # 可視アイテム検出 & リクエスト
    def _schedule_visible_thumbs(self) -> None:
        for item in self._visible_items():
            entry = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(entry, DuplicateClusterEntry):
                # まだプレースホルダならロード要求
                # 既に本物が入っているかの判定を厳密にしない場合は常に要求→inflightで抑止でもOK
                self._queue_thumb(entry.file.path)

    def _visible_items(self) -> list[QTreeWidgetItem]:
        vp = self._tree.viewport()
        h = vp.height()
        y = 0
        items: list[QTreeWidgetItem] = []
        seen = set()
        # indexAt で縦方向に拾う（uniform row heights 前提で軽い）
        while y < h:
            idx = self._tree.indexAt(QPoint(10, y))
            if not idx.isValid():
                break
            it = self._tree.itemFromIndex(idx)
            if it and it not in seen:
                items.append(it)
                seen.add(it)
            rect = self._tree.visualItemRect(it)
            step = rect.height() if rect.height() > 0 else (self._icon_size.height() + 12)
            y += step
        return items

    def _on_thumb_done(self, path_str: str, qimg: "QImage|None") -> None:
        self._thumb_inflight.discard(path_str)
        targets = self._thumb_bindings.get(path_str)
        if not targets:
            self._thumb_done.add(path_str)
            self._maybe_start_more_thumbs()
            return

        try:
            if qimg is not None:
                pix = QPixmap.fromImage(qimg)
            else:
                pix = get_thumbnail(path_str, self._icon_size.width(), self._icon_size.height())
        except Exception:
            pix = None

        for tile in targets:
            tile.set_pixmap(pix, self._placeholder_icon)

        self._thumb_done.add(path_str)
        self._maybe_start_more_thumbs()
        if DEBUG_THUMBS:
            print(f"thumb applied: {path_str}")  # ← デバッグ

    def _update_action_states(self) -> None:
        has_clusters = bool(self._clusters)
        self._mark_button.setEnabled(has_clusters)
        self._uncheck_button.setEnabled(has_clusters)
        self._export_button.setEnabled(has_clusters)
        self._trash_button.setEnabled(self._count_checked() > 0)

    def _on_scan_clicked(self) -> None:
        if self._active_scan is not None:
            return
        self._clusters.clear()
        self._tree.clear()
        self._update_action_states()
        request = self._build_request()
        self._progress.setMaximum(1)
        self._progress.setValue(0)
        self._status_label.setText("Scanning duplicates…")
        runnable = DuplicateScanRunnable(self._db_path, request)
        runnable.signals.progress.connect(self._on_scan_progress)
        runnable.signals.finished.connect(self._on_scan_finished)
        runnable.signals.error.connect(self._on_scan_error)
        self._active_scan = runnable
        self._scan_button.setEnabled(False)
        self._pool.start(runnable)

    def _on_scan_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self._progress.setMaximum(1)
            self._progress.setValue(0)
            return
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _on_scan_finished(self, payload: object) -> None:
        LOG.info("ui: _on_scan_finished begin")
        self._active_scan = None
        self._scan_button.setEnabled(True)
        if not isinstance(payload, list):
            self._status_label.setText("Scan completed with unexpected payload")
            return
        # ★ ここでクラスターをハミング距離の大きい順に並べ替える
        self._clusters = [c for c in payload if isinstance(c, DuplicateCluster)]

        if not self._clusters:
            self._status_label.setText("No duplicate groups detected.")
            self._tree.clear()
            self._update_action_states()
            return
        LOG.info("ui: clusters=%d", len(self._clusters))
        # ← ここで直接 populate せず、まずリファインへ
        self._start_refine_pipeline(self._clusters)
        # リファイン完了後に populate されるので、ここで_populate_tree()は呼ばない

        groups = len(self._clusters)
        files = sum(len(cluster.files) for cluster in self._clusters)
        self._status_label.setText(f"Scan complete: {groups} group(s), {files} file(s).")
        self._update_action_states()
        LOG.info("ui: _on_scan_finished end")

    def _on_scan_error(self, message: str) -> None:
        self._active_scan = None
        self._scan_button.setEnabled(True)
        QMessageBox.critical(self, "Duplicate scan failed", message)
        self._status_label.setText("Duplicate scan failed.")
        self._update_action_states()

    def _build_request(self) -> DuplicateScanRequest:
        ratio = self._ratio_spin.value()
        cosine = self._cosine_spin.value()
        return DuplicateScanRequest(
            path_like=None,
            hamming_threshold=self._hamming_spin.value(),
            size_ratio=ratio if ratio > 0 else None,
            cosine_threshold=cosine if cosine > 0 else None,
        )

    def _group_summary(self, cluster: "DuplicateCluster") -> tuple[str, str, str, str, str]:
        sizes = [e.file.size for e in cluster.files if e.file.size]
        avg_size = int(sum(sizes) / len(sizes)) if sizes else 0

        widths = [e.file.width for e in cluster.files if e.file.width]
        heights = [e.file.height for e in cluster.files if e.file.height]
        avg_w = int(round(sum(widths) / len(widths))) if widths else 0
        avg_h = int(round(sum(heights) / len(heights))) if heights else 0

        hams = [h for h in (e.best_hamming for e in cluster.files) if h is not None]
        avg_ham = (sum(hams) / len(hams)) if hams else None

        coses = [c for c in (e.best_cosine for e in cluster.files) if c is not None]
        avg_cos = (sum(coses) / len(coses)) if coses else None

        first_path = cluster.files[0].file.path.as_posix() if cluster.files else ""

        return (
            self._format_size(avg_size) if avg_size > 0 else "-",
            self._format_resolution(avg_w, avg_h) if (avg_w and avg_h) else "-",
            f"{avg_ham:.1f}" if avg_ham is not None else "-",
            f"{avg_cos:.3f}" if avg_cos is not None else "-",
            first_path,
        )

    def _queue_thumb(self, path: Path) -> None:
        key = str(path)
        if key in self._thumb_inflight or key in self._thumb_done:
            return
        # pending に重複追加しない
        if key not in self._thumb_pending:
            self._thumb_pending.append(key)

    def _panel_of_group(self, group_item: QTreeWidgetItem) -> ThumbPanel | None:
        # 親の直下 0 番目の子にパネルを入れる実装にしている
        if group_item.childCount() == 0:
            return None
        child = group_item.child(0)
        w = self._tree.itemWidget(child, 0)
        return w if isinstance(w, ThumbPanel) else None

    def _populate_tree(self) -> None:
        LOG.info("ui: populate begin")
        # 古い関連づけを掃除（メモリ＆ゴースト参照対策）
        self._thumb_bindings.clear()
        self._thumb_inflight.clear()
        self._thumb_pending.clear()
        self._thumb_done.clear()
        self._block_item_changed = True
        self._bulk_populating = True
        self._tree.setUpdatesEnabled(False)
        self._tree.clear()
        self._populate_i = 0
        self._populate_batch = int(os.environ.get("KE_DUP_POPULATE_BATCH", "300"))

        # タイマーで分割して流し込む
        if hasattr(self, "_populate_timer"):
            self._populate_timer.stop()
        else:
            self._populate_timer = QTimer(self)
            self._populate_timer.setSingleShot(False)
            self._populate_timer.timeout.connect(self._populate_tick)

        self._populate_timer.start(0)  # すぐ1回目

    def _populate_tick(self) -> None:
        n = len(self._clusters)
        start = self._populate_i
        end = min(start + self._populate_batch, n)

        t0 = time.perf_counter()
        self._tree.setUpdatesEnabled(False)
        for i in range(start, end):
            cl = self._clusters[i]
            it = QTreeWidgetItem([""])
            it.setData(0, Qt.ItemDataRole.UserRole, cl)
            # テキストだけ先に入れる（bar は遅延で）
            it.setText(0, f"Group #{i+1} ({len(cl.files)} items)")
            ph = QTreeWidgetItem(["(expand to load)"])
            ph.setData(0, Qt.ItemDataRole.UserRole, "__placeholder__")
            it.addChild(ph)
            it.setExpanded(False)
            self._tree.addTopLevelItem(it)
        self._tree.setUpdatesEnabled(True)

        self._populate_i = end
        dt = time.perf_counter() - t0
        if (end % 1000 == 0) or end == n:
            try:
                import os

                import psutil

                rss = psutil.Process(os.getpid()).memory_info().rss / 1048576.0
                LOG.info("ui: populate %d/%d (%.0fms) RSS=%.1fMB", end, n, dt * 1000, rss)
            except Exception:
                LOG.info("ui: populate %d/%d (%.0fms)", end, n, dt * 1000)

        if end >= n:
            self._populate_timer.stop()
            LOG.info("ui: populate done; groups=%d", self._tree.topLevelItemCount())
            QTimer.singleShot(0, self._expand_initial_groups)

    def _expand_initial_groups(self, approx_px: int | None = None):
        LOG.info("ui: expand_initial")
        vp = self._tree.viewport()
        target_h = approx_px or (vp.height() + 200)
        acc = 0
        for i in range(self._tree.topLevelItemCount()):
            g = self._tree.topLevelItem(i)
            if not g.isExpanded():
                g.setExpanded(True)  # itemExpanded が飛ぶが、下のガードで重い構築はしない
            rect = self._tree.visualItemRect(g)
            acc += rect.height() if rect.height() > 0 else 24
            if acc >= target_h:
                break
        self._thumb_timer.start(0)

    def _on_group_expanded(self, item: QTreeWidgetItem) -> None:
        if self._bulk_populating:
            return
        # ビューに見えている（または直近）ときだけ構築
        self._ensure_panel_built_if_visible(item, force=False)
        self._thumb_timer.start(0)

    def _ensure_panel_built_if_visible(self, item: QTreeWidgetItem, force: bool = False) -> None:
        if self._panel_of_group(item):
            return
        if item.childCount() == 0:
            return
        if item.child(0).data(0, Qt.ItemDataRole.UserRole) != "__placeholder__":
            return

        vp = self._tree.viewport()
        r = self._tree.visualItemRect(item)
        # 画面内±200px を可視扱い（マージン先読み）
        visible = force or (r.isValid() and r.bottom() >= -200 and r.top() <= (vp.height() + 200))
        if not visible:
            return

        # ここで初めて重いパネル生成
        item.takeChildren()
        cluster: DuplicateCluster = item.data(0, Qt.ItemDataRole.UserRole)
        self._build_children_for_cluster(item, cluster)

    def _request_visible_thumbs(self) -> None:
        vp = self._tree.viewport()
        rect = vp.rect()
        wanted: list[str] = []

        for i in range(self._tree.topLevelItemCount()):
            g = self._tree.topLevelItem(i)
            if not g.isExpanded():
                continue

            # 見えていればパネルを用意
            self._ensure_panel_built_if_visible(g, force=False)
            panel = self._panel_of_group(g)
            if not panel:
                continue

            for tile in panel.visible_tiles_in(vp, rect):
                key = str(tile.path)
                if key not in self._thumb_inflight and key not in self._thumb_done:
                    wanted.append(key)

        self._thumb_pending.clear()
        for key in wanted:
            if key not in self._thumb_pending:
                self._thumb_pending.append(key)

        # self._status_label.setText(f"thumb requests (visible): ~{len(self._thumb_pending)}")
        self._maybe_start_more_thumbs()

    # 折りたたみ時は子を捨てて軽量化（必要時にまた作る）
    def _on_group_collapsed(self, item: QTreeWidgetItem) -> None:
        # 古い子のバインディング掃除
        to_unbind: list[str] = []
        for i in range(item.childCount()):
            ch = item.child(i)
            entry = ch.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(entry, DuplicateClusterEntry):
                to_unbind.append(str(entry.file.path))
        for key in to_unbind:
            lst = self._thumb_bindings.get(key)
            if lst:
                self._thumb_bindings[key] = [it for it in lst if it is not None and it.parent() is not None]
                if not self._thumb_bindings[key]:
                    self._thumb_bindings.pop(key, None)
        item.takeChildren()
        placeholder = QTreeWidgetItem(["(expand to load)"])
        placeholder.setData(0, Qt.ItemDataRole.UserRole, "__placeholder__")
        item.addChild(placeholder)

    def _build_children_for_cluster(self, parent_item: QTreeWidgetItem, cluster: "DuplicateCluster") -> None:
        LOG.info("ui: build_panel group_size=%d", len(cluster.files))
        # 並び替えは既存関数をそのまま利用
        entries = self._sort_entries_for_display(cluster.files, cluster.keeper_id)
        panel_item = QTreeWidgetItem(parent_item)
        panel_item.setFirstColumnSpanned(True)
        # パネル生成（アイコン 256 推奨）
        icon256 = QSize(256, 256)
        self._icon_size = icon256  # グリッドに合わせて以後も 256 を使う
        panel = ThumbPanel(entries, cluster.keeper_id, self._icon_size, parent=self._tree)
        self._tree.setItemWidget(panel_item, 0, panel)

        # バインディング
        # 初回＆以降のリサイズで Tree の行高を追従させる
        def _sync_size_hint():
            panel_item.setSizeHint(0, panel.sizeHint())
            self._tree.doItemsLayout()  # F12で飛べないが、executeDelayedItemsLayoutはダメで、doItemsLayoutじゃないといけないらしい
            self._tree.viewport().update()

        panel.sizeHintChanged.connect(_sync_size_hint)
        QTimer.singleShot(0, _sync_size_hint)  # 生成直後にも一度

        for tile in panel.tiles:
            self._bind_tile_to_thumb(tile, tile.path)
            tile.toggled.connect(lambda _=None, self=self: self._update_action_states())
        self._update_action_states()

    def _on_mark_keep_largest(self) -> None:
        for top_idx in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(top_idx)
            if top is None:
                continue
            # 未構築なら展開相当の構築だけ行う（UI上は閉じたまま）
            if top.childCount() == 1 and top.child(0).data(0, Qt.ItemDataRole.UserRole) == "__placeholder__":
                cluster = top.data(0, Qt.ItemDataRole.UserRole)
                top.takeChildren()
                self._build_children_for_cluster(top, cluster)
            # ここから従来通り
            cluster: DuplicateCluster = top.data(0, Qt.ItemDataRole.UserRole)
            for i in range(top.childCount()):
                child = top.child(i)
                entry = child.data(0, Qt.ItemDataRole.UserRole)
                if not isinstance(entry, DuplicateClusterEntry):
                    continue
                state = Qt.CheckState.Unchecked if entry.file.file_id == cluster.keeper_id else Qt.CheckState.Checked
                child.setCheckState(0, state)
        self._schedule_visible_thumbs()
        self._update_action_states()

    def _on_uncheck_all(self) -> None:
        for item, entry in self._iter_tree_entries():
            if entry is None:
                continue
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self._update_action_states()

    def _on_trash_checked(self) -> None:
        checked_entries = [t.entry for t in self._iter_tiles() if t.is_checked()]
        if not checked_entries:
            QMessageBox.information(self, "Trash duplicates", "No files are checked for deletion.")
            return
        successes: list[DuplicateClusterEntry] = []
        failures: list[tuple[DuplicateClusterEntry, str]] = []
        for entry in checked_entries:
            try:
                send2trash(str(entry.file.path))
                successes.append(entry)
            except Exception as exc:  # pragma: no cover - send2trash failures are platform dependent
                failures.append((entry, str(exc)))
        if successes:
            try:
                conn = get_conn(self._db_path)
                try:
                    mark_files_absent(conn, [entry.file.file_id for entry in successes])
                finally:
                    conn.close()
            except Exception as exc:  # pragma: no cover - database errors surfaced via UI
                failures.extend((entry, str(exc)) for entry in successes)
                successes.clear()
        removed_ids = {entry.file.file_id for entry in successes}
        if removed_ids:
            new_clusters: list[DuplicateCluster] = []
            for cluster in self._clusters:
                rebuilt = self._rebuild_cluster(cluster, removed_ids)
                if rebuilt is not None:
                    new_clusters.append(rebuilt)
            self._clusters = new_clusters
            self._populate_tree()
            self._update_status_summary()
        summary = f"Moved {len(successes)} file(s) to trash."
        if failures:
            summary += f" Failed: {len(failures)}."
        self._status_label.setText(summary)
        if failures:
            message = "\n".join(f"{entry.file.path}: {reason}" for entry, reason in failures)
            QMessageBox.warning(self, "Trash duplicates", f"Some files could not be moved:\n{message}")
        self._update_action_states()

    def _rebuild_cluster(self, cluster: DuplicateCluster, removed_ids: set[int]) -> DuplicateCluster | None:
        remaining = [entry for entry in cluster.files if entry.file.file_id not in removed_ids]
        if len(remaining) < 2:
            return None
        new_keeper = self._choose_keeper(remaining)
        sorted_entries = self._sort_entries_for_display(remaining, new_keeper)
        return DuplicateCluster(files=sorted_entries, keeper_id=new_keeper)

    def _choose_keeper(self, entries: Sequence[DuplicateClusterEntry]) -> int:
        def key(entry: DuplicateClusterEntry) -> tuple:
            file = entry.file
            return (
                -(file.size or 0),
                -file.resolution,
                -file.extension_priority,
                file.path.suffix.lower(),
                file.path.name.lower(),
                file.file_id,
            )

        return min(entries, key=key).file.file_id

    def _sort_entries_for_display(
        self, entries: Sequence[DuplicateClusterEntry], keeper_id: int
    ) -> list[DuplicateClusterEntry]:
        return sorted(
            entries,
            key=lambda entry: (
                0 if entry.file.file_id == keeper_id else 1,
                -(entry.file.size or 0),
                -entry.file.resolution,
                -entry.file.extension_priority,
                entry.file.path.name.lower(),
                entry.file.file_id,
            ),
        )

    def _on_export_csv(self) -> None:
        if not self._clusters:
            QMessageBox.information(self, "Export duplicates", "No data to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export duplicate groups", "duplicates.csv", "CSV Files (*.csv)"
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "group",
                        "file_id",
                        "path",
                        "size",
                        "width",
                        "height",
                        "keeper",
                        "hamming",
                        "cosine",
                    ]
                )
                for group_index, cluster in enumerate(self._clusters, start=1):
                    for entry in cluster.files:
                        writer.writerow(
                            [
                                group_index,
                                entry.file.file_id,
                                entry.file.path.as_posix(),
                                entry.file.size or 0,
                                entry.file.width or 0,
                                entry.file.height or 0,
                                1 if entry.file.file_id == cluster.keeper_id else 0,
                                entry.best_hamming if entry.best_hamming is not None else "",
                                f"{entry.best_cosine:.6f}" if entry.best_cosine is not None else "",
                            ]
                        )
        except OSError as exc:  # pragma: no cover - filesystem errors depend on environment
            QMessageBox.warning(self, "Export duplicates", str(exc))
            return
        self._status_label.setText(f"Exported duplicate groups to {file_path}.")

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._block_item_changed or item.parent() is None or column != 0:
            return
        self._update_action_states()

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        entry = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(entry, DuplicateClusterEntry):
            self._reveal_in_file_manager(entry.file.path)

    def _on_context_menu_requested(self, pos: QPoint) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        entry = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(entry, DuplicateClusterEntry):
            return
        menu = QMenu(self)
        open_file_action = QAction("Open file", self)
        open_file_action.triggered.connect(lambda: self._open_path(entry.file.path))
        reveal_action = QAction("Open containing folder", self)
        reveal_action.triggered.connect(lambda: self._reveal_in_file_manager(entry.file.path))
        menu.addAction(open_file_action)
        menu.addAction(reveal_action)
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _open_path(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # pragma: no cover - platform dependent
            QMessageBox.warning(self, "Open file", str(exc))

    def _reveal_in_file_manager(self, path: Path) -> None:
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", str(path)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except Exception as exc:  # pragma: no cover - platform dependent
            QMessageBox.warning(self, "Reveal in folder", str(exc))

    def _iter_tree_entries(self) -> Iterator[tuple[QTreeWidgetItem, DuplicateClusterEntry | None]]:
        for index in range(self._tree.topLevelItemCount()):
            top_item = self._tree.topLevelItem(index)
            for child_index in range(top_item.childCount()):
                child = top_item.child(child_index)
                entry = child.data(0, Qt.ItemDataRole.UserRole)
                yield child, entry if isinstance(entry, DuplicateClusterEntry) else None

    def _count_checked(self) -> int:
        tiles = list(self._iter_tiles())
        if tiles:  # 既に可視タイルがあるときは実際のチェック状態
            return sum(1 for t in tiles if t.is_checked())
        # まだ何も展開していない初期状態は、keeper 以外が Checked 相当
        return sum(max(0, len(c.files) - 1) for c in self._clusters)

    @staticmethod
    def _format_size(value: int | None) -> str:
        if value is None or value <= 0:
            return "-"
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(value)
        index = 0
        while size >= 1024 and index < len(units) - 1:
            size /= 1024
            index += 1
        return f"{size:.1f} {units[index]}"

    @staticmethod
    def _format_resolution(width: int | None, height: int | None) -> str:
        if not width or not height:
            return "-"
        return f"{width}×{height}"


__all__ = ["DupTab"]
