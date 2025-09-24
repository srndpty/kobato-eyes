"""Tests for OpenClipEmbedder handling of varying return signatures."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sig.embedder import OpenClipEmbedder


@pytest.fixture(autouse=True)
def _stub_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a predictable list_pretrained implementation for tests."""

    def _list_pretrained() -> list[tuple[str, str]]:
        return [("ViT-L-14", "openai")]

    monkeypatch.setattr("sig.embedder.open_clip.list_pretrained", _list_pretrained)


def _build_model(output_dim: int = 512) -> MagicMock:
    model = MagicMock()
    model.visual = SimpleNamespace(output_dim=output_dim)
    model.to.return_value = model
    model.eval.return_value = model
    model.half.return_value = model
    return model


def test_openclip_supports_two_tuple_return(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _build_model(320)
    preprocess = object()
    create_mock = MagicMock(return_value=(model, preprocess))
    monkeypatch.setattr(
        "sig.embedder.open_clip.create_model_and_transforms", create_mock
    )

    embedder = OpenClipEmbedder("ViT-L-14", "openai", device="cpu")

    assert embedder.embedding_dim == 320
    assert embedder._preprocess is preprocess  # type: ignore[attr-defined]
    assert create_mock.call_args.kwargs["pretrained"] == "openai"
    assert create_mock.call_args.kwargs["device"].type == "cpu"


def test_openclip_supports_three_tuple_return(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _build_model(640)
    preprocess_train = object()
    preprocess_val = object()
    create_mock = MagicMock(return_value=(model, preprocess_train, preprocess_val))
    monkeypatch.setattr(
        "sig.embedder.open_clip.create_model_and_transforms", create_mock
    )

    embedder = OpenClipEmbedder("ViT-L-14", "openai", device="cpu")

    assert embedder.embedding_dim == 640
    assert embedder._preprocess is preprocess_val  # type: ignore[attr-defined]


def test_openclip_falls_back_when_pretrained_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    model = _build_model(128)
    preprocess = object()
    create_mock = MagicMock(return_value=(model, preprocess))
    monkeypatch.setattr(
        "sig.embedder.open_clip.create_model_and_transforms", create_mock
    )

    def _list_pretrained() -> list[tuple[str, str]]:
        return [("ViT-L-14", "laion2b_s32b_b82k"), ("ViT-L-14", "openai")]

    monkeypatch.setattr("sig.embedder.open_clip.list_pretrained", _list_pretrained)

    with caplog.at_level(logging.INFO):
        embedder = OpenClipEmbedder("ViT-L-14", "nonexistent", device="cpu")

    assert embedder.embedding_dim == 128
    assert create_mock.call_count == 1
    assert create_mock.call_args.kwargs["pretrained"] == "laion2b_s32b_b82k"
    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert any("falling back" in message for message in warning_messages)
    info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert any("laion2b_s32b_b82k" in message for message in info_messages)
