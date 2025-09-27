"""Duplicate detection UI components."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from send2trash import send2trash

from PyQt6.QtCore import QObject, QPoint, Qt, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QHeaderView,
    QHBoxLayout,
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

from core.config import load_settings
from db.connection import get_conn
from db.repository import iter_files_for_dup, mark_files_absent
from dup.scanner import (
    DuplicateCluster,
    DuplicateClusterEntry,
    DuplicateFile,
    DuplicateScanConfig,
    DuplicateScanner,
)
from utils.paths import get_db_path


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
            self.signals.error.emit(str(exc))


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
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
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

        self._update_action_states()

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
        self._clusters = [cluster for cluster in payload if isinstance(cluster, DuplicateCluster)]
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

    def _populate_tree(self) -> None:
        self._block_item_changed = True
        try:
            self._tree.clear()
            for group_index, cluster in enumerate(self._clusters, start=1):
                group_text = f"Group #{group_index} ({len(cluster.files)} items)"
                group_item = QTreeWidgetItem([group_text, "", "", "", "", ""])
                group_item.setFirstColumnSpanned(True)
                flags = group_item.flags()
                flags &= ~Qt.ItemFlag.ItemIsUserCheckable
                group_item.setFlags(flags)
                self._tree.addTopLevelItem(group_item)
                for entry in self._sort_entries_for_display(cluster.files, cluster.keeper_id):
                    item = QTreeWidgetItem(group_item)
                    item.setText(0, entry.file.path.name)
                    item.setText(1, self._format_size(entry.file.size))
                    item.setText(2, self._format_resolution(entry.file.width, entry.file.height))
                    item.setText(3, "-" if entry.best_hamming is None else str(entry.best_hamming))
                    item.setText(
                        4,
                        "-" if entry.best_cosine is None else f"{entry.best_cosine:.3f}",
                    )
                    item.setText(5, entry.file.path.as_posix())
                    item.setData(0, Qt.ItemDataRole.UserRole, entry)
                    state = (
                        Qt.CheckState.Unchecked
                        if entry.file.file_id == cluster.keeper_id
                        else Qt.CheckState.Checked
                    )
                    item.setCheckState(0, state)
                group_item.setExpanded(True)
        finally:
            self._block_item_changed = False
        self._update_action_states()

    def _on_mark_keep_largest(self) -> None:
        for cluster_index, cluster in enumerate(self._clusters):
            top_item = self._tree.topLevelItem(cluster_index)
            if top_item is None:
                continue
            for child_index in range(top_item.childCount()):
                child = top_item.child(child_index)
                entry = child.data(0, Qt.ItemDataRole.UserRole)
                if not isinstance(entry, DuplicateClusterEntry):
                    continue
                state = (
                    Qt.CheckState.Unchecked
                    if entry.file.file_id == cluster.keeper_id
                    else Qt.CheckState.Checked
                )
                child.setCheckState(0, state)
        self._update_action_states()

    def _on_uncheck_all(self) -> None:
        for item, entry in self._iter_tree_entries():
            if entry is None:
                continue
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self._update_action_states()

    def _on_trash_checked(self) -> None:
        checked_entries = [
            entry
            for item, entry in self._iter_tree_entries()
            if entry and item.checkState(0) == Qt.CheckState.Checked
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
        file_path, _ = QFileDialog.getSaveFileName(self, "Export duplicate groups", "duplicates.csv", "CSV Files (*.csv)")
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
            1
            for item, entry in self._iter_tree_entries()
            if entry and item.checkState(0) == Qt.CheckState.Checked
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
