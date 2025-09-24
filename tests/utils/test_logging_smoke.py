"""Smoke tests covering the logging bootstrap helper."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip(
    "PyQt6.QtGui",
    reason="PyQt6 with GUI bindings required for logging smoke test.",
    exc_type=ImportError,
)

from ui.app import setup_logging
from utils.paths import get_log_dir


def _clear_logging_handlers() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def test_setup_logging_creates_rotating_file(tmp_path, monkeypatch) -> None:
    """setup_logging should prepare the log file and accept writes."""

    monkeypatch.setenv("KOE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("KOE_LOG_LEVEL", raising=False)
    _clear_logging_handlers()

    setup_logging()
    logger = logging.getLogger("kobato-eyes.tests")
    message = "logging smoke test"
    logger.info(message)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()

    log_dir = get_log_dir()
    log_path = log_dir / "app.log"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert message in contents
    assert log_path.stat().st_size > 0

