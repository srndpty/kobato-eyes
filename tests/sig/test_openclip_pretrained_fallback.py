"""Tests for OpenClipEmbedder pretrained fallback handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sig.embedder import OpenClipEmbedder


class _DummyModel:
    def __init__(self, output_dim: int = 1024) -> None:
        self.visual = SimpleNamespace(output_dim=output_dim)

    def to(self, device: object) -> None:  # pragma: no cover - interface stub
        self.device = device

    def eval(self) -> None:  # pragma: no cover - interface stub
        return None

    def half(self) -> None:  # pragma: no cover - interface stub
        return None


def _patch_openclip(
    monkeypatch: pytest.MonkeyPatch, available: list[str], recorded: dict[str, str]
) -> None:
    def fake_list_pretrained() -> list[tuple[str, str]]:
        return [("ViT-L-14", tag) for tag in available]

    def fake_create_model_and_transforms(
        model_name: str, *, pretrained: str, device: object | None = None
    ):
        recorded["model_name"] = model_name
        recorded["pretrained"] = pretrained
        recorded["device"] = device
        return _DummyModel(), lambda image: image

    monkeypatch.setattr("sig.embedder.open_clip.list_pretrained", fake_list_pretrained)
    monkeypatch.setattr(
        "sig.embedder.open_clip.create_model_and_transforms", fake_create_model_and_transforms
    )
    monkeypatch.setattr("sig.embedder.torch.cuda.is_available", lambda: False)


def test_fallback_prefers_laion82k(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, str] = {}
    _patch_openclip(monkeypatch, ["laion2b_s32b_b82k", "openai"], recorded)

    embedder = OpenClipEmbedder("ViT-L-14", "bogus", device="cpu")

    assert recorded["model_name"] == "ViT-L-14"
    assert recorded["pretrained"] == "laion2b_s32b_b82k"
    assert embedder.embedding_dim == 1024


def test_fallback_uses_openai_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, str] = {}
    _patch_openclip(monkeypatch, ["openai"], recorded)

    OpenClipEmbedder("ViT-L-14", "bogus", device="cpu")

    assert recorded["pretrained"] == "openai"


def test_fallback_to_first_known_option(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, str] = {}
    _patch_openclip(monkeypatch, ["custom"], recorded)

    OpenClipEmbedder("ViT-L-14", "bogus", device="cpu")

    assert recorded["pretrained"] == "custom"
