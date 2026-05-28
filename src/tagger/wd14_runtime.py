"""ONNX Runtime session helpers for WD14 taggers."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path


def configure_session_options(options, ort_module, logger: logging.Logger) -> None:
    """Apply default optimisation, logging, and profiling settings."""

    options.graph_optimization_level = getattr(ort_module.GraphOptimizationLevel, "ORT_ENABLE_ALL", 99)
    intra_threads = safe_positive_int(os.getenv("KE_ORT_INTRA_OP_THREADS"))
    inter_threads = safe_positive_int(os.getenv("KE_ORT_INTER_OP_THREADS"))
    if intra_threads is not None:
        options.intra_op_num_threads = intra_threads
    if inter_threads is not None:
        options.inter_op_num_threads = inter_threads

    options.enable_profiling = bool(int(os.getenv("KE_ORT_PROFILE", "0")))
    options.log_severity_level = 2
    profile_dir = resolve_profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)
    options.profile_file_prefix = str(profile_dir / "wd14")
    if options.enable_profiling:
        logger.info("WD14: profiling enabled (prefix=%s)", options.profile_file_prefix)


def safe_positive_int(value: str | None) -> int | None:
    """Parse a positive integer environment value."""

    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def log_provider_details(session, chosen: Sequence[str], logger: logging.Logger) -> None:
    """Log provider details for diagnostics in a consistent format."""

    try:
        session_providers = list(session.get_providers())
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("WD14: failed to query session providers: %s", exc)
        session_providers = []
    try:
        provider_options = session.get_provider_options()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("WD14: failed to query provider options: %s", exc)
        provider_options = {}
    logger.info(
        "WD14 providers=%s session_providers=%s options=%s",
        list(chosen),
        session_providers,
        provider_options,
    )


def resolve_profile_dir() -> Path:
    """Resolve the directory used for ONNX Runtime profile output files."""

    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "kobato-eyes" / "logs"
    return Path.home() / "kobato-eyes" / "logs"


__all__ = ["configure_session_options", "log_provider_details", "resolve_profile_dir", "safe_positive_int"]
