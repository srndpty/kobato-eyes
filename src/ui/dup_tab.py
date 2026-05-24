"""Duplicate detection UI components."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Iterator

from PyQt6.QtCore import QEvent, QPoint, QSize, Qt, QThreadPool, QTimer, QtMsgType, qInstallMessageHandler
from PyQt6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.jobs import JobHandle, JobManager, JobPriority
from dup.scanner import DuplicateCluster, DuplicateClusterEntry
from ui.dup_cluster_update import rebuild_clusters_after_removal, sort_entries_for_display
from ui.dup_lifecycle import (
    duplicate_action_availability,
    duplicate_export_status,
    duplicate_refine_cancel_status,
    duplicate_refine_complete_status,
    duplicate_refine_error_status,
    duplicate_refine_progress,
    duplicate_scan_finished_plan,
    duplicate_scan_progress,
    duplicate_trash_summary,
)
from ui.dup_status import format_duplicate_summary
from ui.dup_tree_state import (
    bound_path_from_bindings,
    cluster_hamming_score,
    default_checked_entries,
    should_start_thumbnail,
    unique_pending_thumbnail_keys,
)
from ui.dup_widgets import ThumbPanel, ThumbTile, format_duplicate_resolution, format_duplicate_size
from ui.dup_workers import DuplicateScanJob, DuplicateScanRequest, RefinePipelineJob, ThumbJob, ThumbSignals
from ui.file_actions import export_duplicate_clusters_csv, open_path, reveal_in_file_manager, trash_duplicate_entries
from ui.viewmodels import DupViewModel

logger = logging.getLogger(__name__)
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
    logger.warning(f"QT[{level}] {msg} ({ctx.file}:{ctx.line})")


qInstallMessageHandler(_qt_msg)


class DupTab(QWidget):
    """Provide controls for duplicate search and clustering."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        view_model: DupViewModel | None = None,
    ) -> None:
        super().__init__(parent)
        self._view_model = view_model or DupViewModel(self)
        self._db_path = self._view_model.db_path
        self._jobs = JobManager(max_workers=1, parent=self)
        self._clusters: list[DuplicateCluster] = []
        self._active_scan: JobHandle | None = None
        self._block_item_changed = False
        self._bulk_populating = False
        self._refine_dialog = None
        self._refine_task: JobHandle | None = None

        self._hamming_spin = QSpinBox(self)
        self._hamming_spin.setRange(0, 10)
        self._hamming_spin.setValue(0)

        self._ratio_spin = QDoubleSpinBox(self)
        self._ratio_spin.setRange(0.0, 1.0)
        self._ratio_spin.setSingleStep(0.05)
        self._ratio_spin.setDecimals(2)
        self._ratio_spin.setValue(0.50)
        self._ratio_spin.setSpecialValueText("disabled")

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
        self._idle_expand_timer = QTimer(self)
        self._idle_expand_timer.setSingleShot(True)
        self._idle_expand_timer.setInterval(500)  # ← 0.5秒アイドル
        self._idle_expand_timer.timeout.connect(
            lambda: self._expand_visible_collapsed_groups(max_to_expand=3, margin=120)
        )

        # スクロール“活動”の検知
        def _on_scroll_activity():
            # スクロールが続いている間は常にリセットされ、止まると1秒後に発火
            self._idle_expand_timer.start()

        self._on_scroll_activity = _on_scroll_activity  # メソッドとしてぶら下げてもOK

        # スクロールバー操作で活動扱い
        sb = self._tree.verticalScrollBar()
        sb.valueChanged.connect(lambda _v: (self._thumb_timer.start(30), self._on_scroll_activity()))
        # ついでにドラッグ開始/終了でも保険で活動扱い
        sb.sliderPressed.connect(self._on_scroll_activity)
        sb.sliderReleased.connect(self._on_scroll_activity)

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
        self._thumb_signals = ThumbSignals()
        self._thumb_signals.done.connect(self._on_thumb_done, Qt.ConnectionType.QueuedConnection)

        # ★ サムネイルは可視範囲連動の専用 QThreadPool を使う
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
        self._hb.timeout.connect(lambda: logger.debug(f"ui alive {self._alive:=self._alive+1}"))
        self._hb.start()

    def _update_status_summary(self) -> None:
        self._status_label.setText(format_duplicate_summary(self._clusters))

    def _start_refine_pipeline(self, clusters):
        # パラメータ（好みに応じて UI から取ってもOK）
        tile_params = dict(
            grid=self._grid_spin.value(),
            tile=self._tile_spin.value(),
            max_bits=self._maxbits_spin.value(),
        )
        pixel_params = dict(mae_thr=0.004)

        task = RefinePipelineJob(clusters, tile_params, pixel_params)

        # 進捗ダイアログ
        dlg = QProgressDialog("Refining duplicates...", "Cancel", 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        self._refine_dialog = dlg

        # シグナル接続
        def on_progress(payload):
            cur, total, stage = payload
            state = duplicate_refine_progress(cur, total, stage)
            if state.indeterminate:
                dlg.setRange(0, 0)
            else:
                dlg.setRange(0, state.maximum)
                dlg.setValue(state.value)
            dlg.setLabelText(state.label)

        def on_finished(refined):
            dlg.close()
            self._refine_dialog = None
            self._refine_task = None
            self._scan_button.setEnabled(True)
            # ここで self._clusters を置き換え → 並べ替え → ツリー構築
            self._clusters = refined
            self._clusters.sort(key=lambda c: (self._cluster_hamming_score(c), len(c.files)), reverse=True)
            self._populate_tree()
            self._status_label.setText(duplicate_refine_complete_status(self._clusters))
            self._update_action_states()

        def on_canceled():
            dlg.close()
            self._refine_dialog = None
            self._refine_task = None
            self._scan_button.setEnabled(True)
            self._status_label.setText(duplicate_refine_cancel_status())

        def on_error(msg):
            dlg.close()
            self._refine_dialog = None
            self._refine_task = None
            self._scan_button.setEnabled(True)
            QMessageBox.warning(self, "Refine failed", msg)
            self._status_label.setText(duplicate_refine_error_status())

        handle = self._jobs.submit_handle(task, priority=JobPriority.FOREGROUND)
        self._refine_task = handle
        self._scan_button.setEnabled(False)
        handle.signals.progressState.connect(on_progress)
        handle.signals.completed.connect(on_finished)
        handle.signals.cancelled.connect(on_canceled)
        handle.signals.error.connect(lambda exc, _tb: on_error(str(exc)))

        dlg.canceled.connect(handle.cancel)

        dlg.show()

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
        # ← これを追加：もとの文字列を消して下地の描画を止める
        item.setData(0, Qt.ItemDataRole.DisplayRole, "")  # or item.setText(0, "")

        # ← ついでに高さを同期＆再描画を明示
        item.setSizeHint(0, bar.sizeHint())
        self._tree.viewport().update(self._tree.visualItemRect(item))

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
                pix = self._view_model.get_thumbnail(Path(key), self._icon_size.width(), self._icon_size.height())
                tile.set_pixmap(pix, self._placeholder_icon)
            except Exception:
                tile.set_pixmap(None, self._placeholder_icon)

    def _maybe_start_more_thumbs(self) -> None:
        slots = self._thumb_pool.maxThreadCount() - self._thumb_pool.activeThreadCount()
        # 1 ティックの上限（スクロール直後のスパイク抑止）
        slots = min(slots, self._thumb_budget)
        cache_dir = self._view_model.thumbnail_cache_dir()
        size = (self._icon_size.width(), self._icon_size.height())

        while slots > 0 and self._thumb_pending:
            key = self._thumb_pending.popleft()
            if not should_start_thumbnail(key, inflight=self._thumb_inflight, done=self._thumb_done):
                continue
            path = Path(key)
            job = ThumbJob(self._view_model, path, size, cache_dir, self._thumb_signals)
            self._thumb_inflight.add(key)
            self._thumb_pool.start(job)
            slots -= 1

    def _bound_path_of_item(self, item: QTreeWidgetItem) -> Path | None:
        # 可視行は数十件なので O(可視件数) の探索で十分軽い
        return bound_path_from_bindings(self._thumb_bindings, item)

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
                if should_start_thumbnail(key, inflight=self._thumb_inflight, done=self._thumb_done):
                    wanted.append(key)

        self._thumb_pending.clear()
        self._thumb_pending.extend(
            unique_pending_thumbnail_keys(wanted, inflight=self._thumb_inflight, done=self._thumb_done)
        )

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

    def _iter_checked_entries(self) -> Iterator[DuplicateClusterEntry]:
        """Yield entries that are currently marked for deletion."""

        for idx in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(idx)
            if top is None:
                continue
            cluster = top.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(cluster, DuplicateCluster):
                continue

            panel = self._panel_of_group(top)
            if panel is not None:
                for tile in panel.tiles:
                    if tile.is_checked():
                        yield tile.entry
                continue

            # 未展開のクラスターは keeper 以外が Checked 相当
            for entry in default_checked_entries(cluster):
                yield entry

    def _cluster_hamming_score(self, cluster: "DuplicateCluster") -> int:
        """
        クラスター内で keeper 以外の best_hamming の最大値を返す。
        値が大きいほど“目視確認の必要性が高い”とみなす。
        すべて None の場合は -1（最下位扱い）を返す。
        """
        return cluster_hamming_score(cluster)

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
                pix = self._view_model.get_thumbnail(Path(key), self._icon_size.width(), self._icon_size.height())
                item.setIcon(0, QIcon(pix))
            except Exception:
                item.setIcon(0, self._placeholder_icon)

    def _request_thumb(self, path: Path) -> None:
        key = str(path)
        if key in self._thumb_inflight:
            return
        self._thumb_inflight.add(key)
        cache_dir = self._view_model.thumbnail_cache_dir()
        if DEBUG_THUMBS:
            logger.info(f"thumb enqueue: {key}")  # ← デバッグ
        job = ThumbJob(
            self._view_model,
            path,
            (self._icon_size.width(), self._icon_size.height()),
            cache_dir,
            self._thumb_signals,
        )
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
                self._on_scroll_activity()

            if ev.type() == QEvent.Type.Wheel:  # ← 追加
                self._thumb_timer.start(0)
                self._on_scroll_activity()

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

    def _expand_visible_collapsed_groups(self, max_to_expand: int = 3, margin: int = 120) -> None:
        # 大量追加中はスキップ（描画が落ち着いてから）
        if getattr(self, "_bulk_populating", False):
            return

        vp = self._tree.viewport()
        rect = vp.rect().adjusted(0, -margin, 0, +margin)

        expanded = 0
        # 画面上から可視範囲を舐める
        idx = self._tree.indexAt(QPoint(10, 10))
        while idx.isValid():
            it = self._tree.itemFromIndex(idx)
            if it is None:
                break
            if it.parent() is None:  # トップレベル
                r = self._tree.visualItemRect(it)
                if r.bottom() < rect.top():
                    idx = self._tree.indexBelow(idx)
                    continue
                if r.top() > rect.bottom():
                    break

                if not it.isExpanded():
                    # 先に軽量ヘッダーを（未装着なら）付与してから展開
                    self._ensure_header_built_if_visible(it)
                    it.setExpanded(True)  # → _on_group_expanded で panel を遅延構築
                    expanded += 1
                    if expanded >= max_to_expand:
                        break

                # すでに展開済みなら、必要ならパネルも可視条件で構築
                else:
                    self._ensure_panel_built_if_visible(it, force=False)

            idx = self._tree.indexBelow(idx)

        if expanded:
            # サムネ要求を即キック
            self._thumb_timer.start(0)

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
                pix = self._view_model.get_thumbnail(Path(path_str), self._icon_size.width(), self._icon_size.height())
        except Exception:
            pix = None

        for tile in targets:
            tile.set_pixmap(pix, self._placeholder_icon)

        self._thumb_done.add(path_str)
        self._maybe_start_more_thumbs()
        if DEBUG_THUMBS:
            logger.info(f"thumb applied: {path_str}")  # ← デバッグ

    def _update_action_states(self) -> None:
        availability = duplicate_action_availability(
            has_clusters=bool(self._clusters),
            checked_count=self._count_checked(),
        )
        self._mark_button.setEnabled(availability.mark)
        self._uncheck_button.setEnabled(availability.uncheck)
        self._export_button.setEnabled(availability.export)
        self._trash_button.setEnabled(availability.trash)

    def _on_scan_clicked(self) -> None:
        if self._active_scan is not None or self._refine_task is not None:
            return
        self._clusters.clear()
        self._tree.clear()
        self._update_action_states()
        request = self._build_request()
        self._progress.setMaximum(1)
        self._progress.setValue(0)
        self._status_label.setText("Scanning duplicates...")
        job = DuplicateScanJob(self._view_model, self._db_path, request)
        handle = self._jobs.submit_handle(job, priority=JobPriority.FOREGROUND)
        handle.signals.progressState.connect(lambda payload: self._on_scan_progress_state(*payload))
        handle.signals.completed.connect(self._on_scan_finished)
        handle.signals.error.connect(lambda exc, _tb: self._on_scan_error(str(exc)))
        self._active_scan = handle
        self._scan_button.setEnabled(False)

    def _on_scan_progress(self, current: int, total: int) -> None:
        state = duplicate_scan_progress(current, total)
        self._apply_scan_progress_state(state)

    def _on_scan_progress_state(self, current: int, total: int, stage: str) -> None:
        """Apply duplicate scan progress with stage context."""

        state = duplicate_scan_progress(current, total, stage)
        self._apply_scan_progress_state(state)

    def _apply_scan_progress_state(self, state) -> None:
        if state.indeterminate:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, state.maximum)
            self._progress.setValue(state.value)
        self._status_label.setText(state.label)

    def _on_scan_finished(self, payload: object) -> None:
        logger.info("ui: _on_scan_finished begin")
        self._active_scan = None
        self._scan_button.setEnabled(True)
        plan = duplicate_scan_finished_plan(payload, DuplicateCluster)
        self._status_label.setText(plan.status)
        if not plan.valid_payload:
            return
        self._clusters = [cluster for cluster in plan.clusters if isinstance(cluster, DuplicateCluster)]
        if plan.clear_tree:
            self._tree.clear()
            self._update_action_states()
            return
        logger.info("ui: clusters=%d", len(self._clusters))
        if plan.refine:
            self._start_refine_pipeline(self._clusters)
        self._update_action_states()
        logger.info("ui: _on_scan_finished end")

    def _on_scan_error(self, message: str) -> None:
        self._active_scan = None
        self._scan_button.setEnabled(True)
        QMessageBox.critical(self, "Duplicate scan failed", message)
        self._status_label.setText("Duplicate scan failed.")
        self._update_action_states()

    def _build_request(self) -> DuplicateScanRequest:
        ratio = self._ratio_spin.value()
        return DuplicateScanRequest(
            path_like=None,
            hamming_threshold=self._hamming_spin.value(),
            size_ratio=ratio if ratio > 0 else None,
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
        logger.info("ui: populate begin")
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
            it.setText(0, f"Group #{i + 1} ({len(cl.files)} items)")
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
                logger.info("ui: populate %d/%d (%.0fms) RSS=%.1fMB", end, n, dt * 1000, rss)
            except Exception:
                logger.info("ui: populate %d/%d (%.0fms)", end, n, dt * 1000)

        if end >= n:
            self._populate_timer.stop()
            self._bulk_populating = False
            self._block_item_changed = False
            self._tree.setUpdatesEnabled(True)
            logger.info("ui: populate done; groups=%d", self._tree.topLevelItemCount())
            QTimer.singleShot(0, self._expand_initial_groups)
            if hasattr(self, "_idle_expand_timer"):
                self._idle_expand_timer.start()

    def _expand_initial_groups(self, approx_px: int | None = None):
        logger.info("ui: expand_initial")
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
        # logger.info("ui: build_panel group_size=%d", len(cluster.files))
        # 並び替えは既存関数をそのまま利用
        entries = sort_entries_for_display(cluster.files, cluster.keeper_id)
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
        checked_entries = list(self._iter_checked_entries())
        if not checked_entries:
            QMessageBox.information(self, "Trash duplicates", "No files are checked for deletion.")
            return
        successes, failures = trash_duplicate_entries(checked_entries)
        if successes:
            try:
                with closing(self._view_model.open_connection(self._db_path)) as conn:
                    self._view_model.mark_files_absent(conn, [entry.file.file_id for entry in successes])
            except Exception as exc:  # pragma: no cover - database errors surfaced via UI
                failures.extend((entry, str(exc)) for entry in successes)
                successes.clear()
        removed_ids = {entry.file.file_id for entry in successes}
        if removed_ids:
            self._clusters = rebuild_clusters_after_removal(self._clusters, removed_ids)
            self._populate_tree()
            self._update_status_summary()
        self._status_label.setText(duplicate_trash_summary(len(successes), len(failures)))
        if failures:
            message = "\n".join(f"{entry.file.path}: {reason}" for entry, reason in failures)
            QMessageBox.warning(self, "Trash duplicates", f"Some files could not be moved:\n{message}")
        self._update_action_states()

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
            export_duplicate_clusters_csv(self._clusters, file_path)
        except OSError as exc:  # pragma: no cover - filesystem errors depend on environment
            QMessageBox.warning(self, "Export duplicates", str(exc))
            return
        self._status_label.setText(duplicate_export_status(file_path))

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
            open_path(path)
        except Exception as exc:  # pragma: no cover - platform dependent
            QMessageBox.warning(self, "Open file", str(exc))

    def _reveal_in_file_manager(self, path: Path) -> None:
        try:
            reveal_in_file_manager(path)
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
        return sum(1 for _ in self._iter_checked_entries())

    @staticmethod
    def _format_size(value: int | None) -> str:
        return format_duplicate_size(value)

    @staticmethod
    def _format_resolution(width: int | None, height: int | None) -> str:
        return format_duplicate_resolution(width, height)


__all__ = ["DupTab"]
