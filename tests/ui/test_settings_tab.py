"""Smoke tests for SettingsTab background job orchestration."""

from __future__ import annotations

from typing import Iterable

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 widgets required", exc_type=ImportError)
from PyQt6.QtGui import QHideEvent
from PyQt6.QtWidgets import QApplication

from core.config import PipelineSettings
from ui.settings_tab import SettingsTab
from ui.viewmodels import SettingsViewModel

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Iterable[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _Signals:
    def __init__(self) -> None:
        self.completed = _Signal()
        self.cancelled = _Signal()
        self.error = _Signal()


class _Handle:
    def __init__(self) -> None:
        self.signals = _Signals()
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _InspectionJobs:
    def __init__(self) -> None:
        self.handles: list[_Handle] = []
        self.jobs: list[object] = []

    def submit_handle(self, job, priority) -> _Handle:
        del priority
        handle = _Handle()
        self.jobs.append(job)
        self.handles.append(handle)
        return handle


def test_model_inspection_cancels_stale_handle(qapp: QApplication) -> None:
    view_model = SettingsViewModel(provider_loader=lambda: [])
    widget = SettingsTab(view_model=view_model)
    jobs = _InspectionJobs()
    widget._inspection_jobs = jobs  # type: ignore[attr-defined]

    try:
        widget._inspection_generation = 1  # type: ignore[attr-defined]
        widget._start_model_inspection()  # type: ignore[attr-defined]
        first = jobs.handles[0]
        assert widget._inspection_task is first  # type: ignore[attr-defined]

        widget._inspection_generation = 2  # type: ignore[attr-defined]
        widget._start_model_inspection()  # type: ignore[attr-defined]

        assert first.cancelled is True
        assert widget._inspection_task is jobs.handles[1]  # type: ignore[attr-defined]

        jobs.handles[1].signals.completed.emit((2, "Model status: OK", True))
        assert widget._inspection_task is None  # type: ignore[attr-defined]
    finally:
        widget.deleteLater()
        qapp.processEvents()


def test_settings_tab_auto_applies_pending_changes_on_hide(qapp: QApplication) -> None:
    view_model = SettingsViewModel(provider_loader=lambda: [])
    view_model.set_current_settings(PipelineSettings(batch_size=8, prefetch_depth=4))
    widget = SettingsTab(view_model=view_model)
    applied: list[PipelineSettings] = []
    widget.settings_applied.connect(applied.append)

    try:
        widget._batch_size_spin.setValue(12)  # type: ignore[attr-defined]
        widget._prefetch_depth_spin.setValue(7)  # type: ignore[attr-defined]
        widget._tagger_input_cache_check.setChecked(True)  # type: ignore[attr-defined]

        widget.hideEvent(QHideEvent())  # type: ignore[attr-defined]
        qapp.processEvents()

        assert applied
        assert applied[-1].batch_size == 12
        assert applied[-1].prefetch_depth == 7
        assert applied[-1].tagger_input_cache is True
        assert widget._dirty is False  # type: ignore[attr-defined]
    finally:
        widget.deleteLater()
        qapp.processEvents()
