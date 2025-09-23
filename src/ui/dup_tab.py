"""Duplicate detection UI components."""

from __future__ import annotations

from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class DupTab(QWidget):
    """Provide controls for duplicate search and clustering."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file_input = QLineEdit(self)
        self._file_input.setPlaceholderText("Enter file path or ID…")
        self._search_button = QPushButton("Search", self)
        self._cluster_button = QPushButton("Cluster All", self)
        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)

        self._hamming_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._hamming_slider.setRange(0, 64)
        self._hamming_slider.setValue(8)
        self._hamming_value = QLabel("8", self)

        self._cosine_spin = QDoubleSpinBox(self)
        self._cosine_spin.setRange(0.0, 1.0)
        self._cosine_spin.setSingleStep(0.01)
        self._cosine_spin.setValue(0.2)

        self._ssim_spin = QDoubleSpinBox(self)
        self._ssim_spin.setRange(0.0, 1.0)
        self._ssim_spin.setSingleStep(0.01)
        self._ssim_spin.setValue(0.9)

        self._candidate_table = QTableWidget(0, 3, self)
        self._candidate_table.setHorizontalHeaderLabels(["Candidate", "pHash", "Cosine"])
        self._candidate_table.horizontalHeader().setStretchLastSection(True)
        self._candidate_table.verticalHeader().setVisible(False)
        self._candidate_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self._cluster_table = QTableWidget(0, 2, self)
        self._cluster_table.setHorizontalHeaderLabels(["Representative", "Members"])
        self._cluster_table.horizontalHeader().setStretchLastSection(True)
        self._cluster_table.verticalHeader().setVisible(False)
        self._cluster_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Hamming ≤", self))
        threshold_layout.addWidget(self._hamming_slider)
        threshold_layout.addWidget(self._hamming_value)
        threshold_layout.addWidget(QLabel("Cosine ≤", self))
        threshold_layout.addWidget(self._cosine_spin)
        threshold_layout.addWidget(QLabel("SSIM ≥", self))
        threshold_layout.addWidget(self._ssim_spin)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._search_button)
        button_layout.addWidget(self._cluster_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._file_input)
        layout.addLayout(threshold_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self._status_label)
        layout.addWidget(QLabel("Candidates", self))
        layout.addWidget(self._candidate_table)
        layout.addWidget(QLabel("Clusters", self))
        layout.addWidget(self._cluster_table)

        self._hamming_slider.valueChanged.connect(self._on_hamming_changed)
        self._search_button.clicked.connect(self._on_search_clicked)
        self._cluster_button.clicked.connect(self._on_cluster_clicked)

    def _on_hamming_changed(self, value: int) -> None:
        self._hamming_value.setText(str(value))

    def _on_search_clicked(self) -> None:
        query = self._file_input.text().strip() or "(none)"
        message = (
            f"Searching {query} with hamming≤{self._hamming_slider.value()}, "
            f"cosine≤{self._cosine_spin.value():.2f}, ssim≥{self._ssim_spin.value():.2f}"
        )
        self._status_label.setText(message)
        self._candidate_table.setRowCount(0)

    def _on_cluster_clicked(self) -> None:
        message = f"Clustering with cosine≤{self._cosine_spin.value():.2f}, " f"ssim≥{self._ssim_spin.value():.2f}"
        self._status_label.setText(message)
        self._cluster_table.setRowCount(0)

    def display_candidates(self, rows: Iterable[tuple[int, int | None, float | None]]) -> None:
        """Populate candidate table with (file_id, phash_distance, cosine_distance)."""
        self._candidate_table.setRowCount(0)
        for index, (file_id, phash_distance, cosine_distance) in enumerate(rows):
            self._candidate_table.insertRow(index)
            self._candidate_table.setItem(index, 0, QTableWidgetItem(str(file_id)))
            self._candidate_table.setItem(
                index, 1, QTableWidgetItem("-" if phash_distance is None else str(phash_distance))
            )
            cos_text = "-" if cosine_distance is None else f"{cosine_distance:.4f}"
            self._candidate_table.setItem(index, 2, QTableWidgetItem(cos_text))

    def display_clusters(self, rows: Iterable[tuple[int, list[int]]]) -> None:
        """Populate cluster table with representative and members."""
        self._cluster_table.setRowCount(0)
        for index, (representative, members) in enumerate(rows):
            self._cluster_table.insertRow(index)
            self._cluster_table.setItem(index, 0, QTableWidgetItem(str(representative)))
            member_text = ", ".join(str(member) for member in members)
            self._cluster_table.setItem(index, 1, QTableWidgetItem(member_text))


__all__ = ["DupTab"]
