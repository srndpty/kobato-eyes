"""Structured exception types for kobato-eyes service boundaries."""

from __future__ import annotations


class PipelineError(Exception):
    """パイプラインステージ（scan/tag/write）の失敗を表す。"""


class DBServiceError(Exception):
    """DBWritingService のワーカースレッド障害を表す。"""


class SignatureComputeError(Exception):
    """署名計算（fastsig）の失敗または中断を表す。"""
