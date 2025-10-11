"""Tests for environment helpers controlling headless behaviour."""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

from utils import env


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_is_headless_false_for_falsey_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """is_headless() should reject typical falsey environment values."""
    monkeypatch.setenv("KOE_HEADLESS", value)
    assert env.is_headless() is False


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_is_headless_true_for_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Ensure is_headless() recognises truthy flag values."""
    monkeypatch.setenv("KOE_HEADLESS", value)
    assert env.is_headless() is True


def test_jobs_define_dummy_qobject_when_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload core.jobs with headless flag to exercise dummy Qt shims."""
    pipeline_stub = types.ModuleType("core.pipeline")
    monkeypatch.setitem(sys.modules, "core.pipeline", pipeline_stub)

    original_value = os.environ.get("KOE_HEADLESS")
    monkeypatch.setenv("KOE_HEADLESS", "1")
    jobs_module = importlib.reload(importlib.import_module("core.jobs"))

    assert hasattr(jobs_module.QObject, "_push_sender")
    assert jobs_module.QObject.__module__ == "core.jobs"

    if original_value is None:
        monkeypatch.delenv("KOE_HEADLESS", raising=False)
    else:
        monkeypatch.setenv("KOE_HEADLESS", original_value)

    try:
        importlib.reload(jobs_module)
    except ImportError:
        pass
