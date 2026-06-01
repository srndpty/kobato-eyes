"""Tests for structured exception types in core.exceptions."""

from __future__ import annotations

from core.exceptions import DBServiceError, PipelineError, SignatureComputeError


def test_pipeline_error_is_exception() -> None:
    exc = PipelineError("scan failed")
    assert isinstance(exc, Exception)
    assert "scan failed" in str(exc)


def test_db_service_error_is_exception() -> None:
    exc = DBServiceError("worker crashed")
    assert isinstance(exc, Exception)
    assert "worker crashed" in str(exc)


def test_signature_compute_error_is_exception() -> None:
    exc = SignatureComputeError("cancelled")
    assert isinstance(exc, Exception)
    assert "cancelled" in str(exc)
