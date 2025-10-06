"""Duplicate detection UI components."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QEvent, QObject, QPoint, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPixmap  # 既にあれば不要
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
from utils.image_io import generate_thumbnail, get_thumbnail  # 既存ユーティリティを再利用
from utils.paths import get_cache_dir, get_db_path

DEBUG_THUMBS = False


@dataclass(frozen=True)
class DuplicateScanRequest:
    """Parameters supplied to the duplicate scanning worker."""

    path_like: str | None
    hamming_threshold: int
    size_ratio: float | None
    cosine_threshold: float | None


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

    def run(self) -> None:  # noqa: D401 - QRunnable interface
        try:
            settings = load_settings()
            model_name = settings.model_name
            conn = get_conn(self._db_path)
            try:
                rows = list(iter_files_for_dup(conn, self._request.path_like, model_name=model_name))
            finally:
                conn.close()
            total = len(rows)
            self.signals.progress.emit(0, total)
            files: list[DuplicateFile] = []
            for index, row in enumerate(rows, start=1):
                try:
                    files.append(DuplicateFile.from_row(row))
                except ValueError:
                    continue
                self.signals.progress.emit(index, total)
            config = DuplicateScanConfig(
                hamming_threshold=self._request.hamming_threshold,
                size_ratio=self._request.size_ratio,
                cosine_threshold=self._request.cosine_threshold,
            )
            clusters = DuplicateScanner(config).build_clusters(files)
            self.signals.finished.emit(clusters)
        except Exception as exc:  # pragma: no cover - surfaced via UI
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

        self._path_input = QLineEdit(self)
        self._path_input.setPlaceholderText("Path LIKE pattern (e.g. C:% or %/images/% )")

        self._hamming_spin = QSpinBox(self)
        self._hamming_spin.setRange(0, 64)
        self._hamming_spin.setValue(8)

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

        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(6)
        self._tree.setHeaderLabels(["File", "Size", "Resolution", "Hamming", "Cosine", "Path"])
        self._tree.setAlternatingRowColors(True)
        # ★ アイコンサイズ（Tagsタブに寄せるなら 96x96 や 128x128）
        self._icon_size = QSize(96, 96)
        self._tree.setIconSize(self._icon_size)
        # 行高をアイコン＋余白に固定（UniformRowHeights と相性◎）
        row_h = self._icon_size.height() + 8
        self._tree.setUniformRowHeights(True)
        self._tree.setStyleSheet(f"QTreeView::item {{ height: {row_h}px; padding: 2px; }}")

        # サムネの可視範囲リクエストをタイマでデバウンス
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.timeout.connect(self._request_visible_thumbs)
        self._tree.verticalScrollBar().valueChanged.connect(lambda _: self._thumb_timer.start(30))
        self._tree.viewport().installEventFilter(self)

        # サムネ用スレッドプールを控えめに（I/O スパイク防止）
        self._pool.setMaxThreadCount(min(4, os.cpu_count() or 2))

        header = self._tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # ← 全列を手動幅に
        # だいたいの初期幅（必要なら好みで調整）
        header.resizeSection(0, self._icon_size.width() + 220)  # File(+サムネ)
        header.resizeSection(1, 80)  # Size
        header.resizeSection(2, 90)  # Resolution
        header.resizeSection(3, 80)  # Hamming
        header.resizeSection(4, 80)  # Cosine
        header.resizeSection(5, 420)  # Path
        header.setStretchLastSection(True)

        self._tree.setUniformRowHeights(True)
        self._tree.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self._tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self._tree.setAnimated(False)
        self._tree.setSortingEnabled(False)
        self._tree.setExpandsOnDoubleClick(False)

        self._tree.setRootIsDecorated(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu_requested)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Path LIKE", self))
        controls_layout.addWidget(self._path_input, 1)
        controls_layout.addWidget(QLabel("Hamming ≤", self))
        controls_layout.addWidget(self._hamming_spin)
        controls_layout.addWidget(QLabel("Size ratio ≥", self))
        controls_layout.addWidget(self._ratio_spin)
        controls_layout.addWidget(QLabel("Cosine ≤", self))
        controls_layout.addWidget(self._cosine_spin)

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
        self._thumb_bindings: dict[str, list[QTreeWidgetItem]] = {}

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
        vp = self._tree.viewport()
        if self._tree.topLevelItemCount() == 0:
            return

        # 可視最上 index（未計算保険）
        top_idx = self._tree.indexAt(vp.rect().topLeft())
        if not top_idx.isValid():
            top_idx = self._tree.model().index(0, 0)

        # 上に少し戻す（先読み）
        idx = top_idx
        for _ in range(self._prefetch_rows):
            prev = self._tree.indexAbove(idx)
            if not prev.isValid():
                break
            idx = prev

        bottom_y = vp.rect().bottom()

        wanted: list[str] = []
        seen_keys: set[str] = set()
        after_bottom_rows = self._prefetch_rows

        # 可視＋マージンの範囲でキー収集（上→下の順序）
        while idx.isValid():
            rect = self._tree.visualRect(idx)
            if rect.top() > bottom_y:
                if after_bottom_rows <= 0:
                    break
                after_bottom_rows -= 1

            item = self._tree.itemFromIndex(idx)
            if item is not None:
                # まずプレースホルダ（軽い）
                if item.icon(0).isNull():
                    item.setIcon(0, self._placeholder_icon)
                # この item にバインドされている path を逆引き
                path = self._bound_path_of_item(item)
                if path is not None:
                    key = str(path)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        wanted.append(key)

            idx = self._tree.indexBelow(idx)

        # ---- pending を「wanted の順序で」再構成（ここで可視外は捨てられる） ----
        self._thumb_pending.clear()
        for key in wanted:
            if key not in self._thumb_inflight and key not in self._thumb_done:
                self._thumb_pending.append(key)

        # ステータス（任意）
        self._status_label.setText(f"thumb requests (visible): ~{len(self._thumb_pending)}")

        # 空きスロット分だけ起動
        self._maybe_start_more_thumbs()

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
        items = self._thumb_bindings.get(path_str)
        if not items:
            return
        try:
            if qimg is not None:
                pix = QPixmap.fromImage(qimg)
            else:
                # 最悪時だけ UI で読み込み（頻度は激減）
                pix = get_thumbnail(path_str, self._icon_size.width(), self._icon_size.height())
            icon = QIcon(pix)
        except Exception:
            icon = self._placeholder_icon

        self._tree.setUpdatesEnabled(False)
        for it in items:
            it.setIcon(0, icon)
        self._tree.setUpdatesEnabled(True)
        self._thumb_done.add(path_str)
        self._maybe_start_more_thumbs()  # 次の pending を流し込み
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
        self._active_scan = None
        self._scan_button.setEnabled(True)
        if not isinstance(payload, list):
            self._status_label.setText("Scan completed with unexpected payload")
            return
        # ★ ここでクラスターをハミング距離の大きい順に並べ替える
        self._clusters = [c for c in payload if isinstance(c, DuplicateCluster)]
        # 同点は「グループサイズが大きいほう」を優先したい場合はタプルキーにする
        self._clusters.sort(key=lambda c: (self._cluster_hamming_score(c), len(c.files)), reverse=True)

        if not self._clusters:
            self._status_label.setText("No duplicate groups detected.")
            self._tree.clear()
            self._update_action_states()
            return
        self._populate_tree()
        groups = len(self._clusters)
        files = sum(len(cluster.files) for cluster in self._clusters)
        self._status_label.setText(f"Scan complete: {groups} group(s), {files} file(s).")
        self._update_action_states()

    def _on_scan_error(self, message: str) -> None:
        self._active_scan = None
        self._scan_button.setEnabled(True)
        QMessageBox.critical(self, "Duplicate scan failed", message)
        self._status_label.setText("Duplicate scan failed.")
        self._update_action_states()

    def _build_request(self) -> DuplicateScanRequest:
        path_text = self._path_input.text().strip()
        like = None
        if path_text:
            like = path_text if any(ch in path_text for ch in "%_") else f"{path_text}%"
        ratio = self._ratio_spin.value()
        cosine = self._cosine_spin.value()
        return DuplicateScanRequest(
            path_like=like,
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

    # 既存 _populate_tree を「トップレベルだけ作る」版に差し替え
    def _populate_tree(self) -> None:
        # 古い関連づけを掃除（メモリ＆ゴースト参照対策）
        self._thumb_bindings.clear()
        self._thumb_inflight.clear()
        self._thumb_pending.clear()
        self._thumb_done.clear()
        self._block_item_changed = True
        try:
            self._tree.setUpdatesEnabled(False)
            self._tree.clear()

            for group_index, cluster in enumerate(self._clusters, start=1):
                group_text = f"Group #{group_index} ({len(cluster.files)} items)"
                group_item = QTreeWidgetItem(["", "", "", "", "", ""])
                group_item.setText(0, group_text)
                group_item.setData(0, Qt.ItemDataRole.UserRole, cluster)
                group_item.setFirstColumnSpanned(False)  # ← スパンをやめて各列にサマリを表示

                # 親サムネ: 先頭アイテムにバインドして非同期読み込み
                if cluster.files:
                    first_path = cluster.files[0].file.path
                    group_item.setIcon(0, self._placeholder_icon)
                    self._bind_item_to_thumb(group_item, first_path)

                # サマリ値を各列に
                size_t, res_t, ham_t, cos_t, path_t = self._group_summary(cluster)
                group_item.setText(1, size_t)
                group_item.setText(2, res_t)
                group_item.setText(3, ham_t)
                group_item.setText(4, cos_t)
                group_item.setText(5, path_t)

                # プレースホルダ子（遅延構築フラグ）
                placeholder = QTreeWidgetItem(["(expand to load)"])
                placeholder.setData(0, Qt.ItemDataRole.UserRole, "__placeholder__")
                group_item.addChild(placeholder)

                self._tree.addTopLevelItem(group_item)
                group_item.setExpanded(False)
        finally:
            self._tree.setUpdatesEnabled(True)
            self._block_item_changed = False
        self._update_action_states()
        self._tree.executeDelayedItemsLayout()
        self._tree.viewport().update()
        QTimer.singleShot(0, self._request_visible_thumbs)  # 1フレーム後に可視範囲要求

    # 展開時にだけ子を構築
    def _on_group_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() == 1 and item.child(0).data(0, Qt.ItemDataRole.UserRole) == "__placeholder__":
            item.takeChildren()
            cluster: DuplicateCluster = item.data(0, Qt.ItemDataRole.UserRole)
            self._build_children_for_cluster(item, cluster)
        self._thumb_timer.start(0)

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

    # 既存ロジックを流用して“子だけ”作る関数
    def _build_children_for_cluster(self, parent: QTreeWidgetItem, cluster: "DuplicateCluster") -> None:
        self._tree.setUpdatesEnabled(False)
        try:
            for entry in self._sort_entries_for_display(cluster.files, cluster.keeper_id):
                item = QTreeWidgetItem(parent)
                item.setText(0, entry.file.path.name)
                item.setText(1, self._format_size(entry.file.size))
                item.setText(2, self._format_resolution(entry.file.width, entry.file.height))
                item.setText(3, "-" if entry.best_hamming is None else str(entry.best_hamming))
                item.setText(4, "-" if entry.best_cosine is None else f"{entry.best_cosine:.3f}")
                item.setText(5, entry.file.path.as_posix())
                item.setData(0, Qt.ItemDataRole.UserRole, entry)
                # プレースホルダを先につける（本物は可視時に）
                item.setIcon(0, self._placeholder_icon)
                # チェック状態
                state = Qt.CheckState.Unchecked if entry.file.file_id == cluster.keeper_id else Qt.CheckState.Checked
                item.setCheckState(0, state)
                # サムネ適用対象にバインド（実際のロードは可視範囲だけ後段で）
                self._bind_item_to_thumb(item, entry.file.path)
        finally:
            self._tree.setUpdatesEnabled(True)

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
        checked_entries = [
            entry for item, entry in self._iter_tree_entries() if entry and item.checkState(0) == Qt.CheckState.Checked
        ]
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
        return sum(
            1 for item, entry in self._iter_tree_entries() if entry and item.checkState(0) == Qt.CheckState.Checked
        )

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
