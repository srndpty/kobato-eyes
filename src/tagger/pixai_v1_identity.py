"""Content identities for PixAI v1 retagging decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.schema import TaggerSettings

from tagger.base import MaxTagsMap, ThresholdMap
from tagger.pixai_v1_labels import DEFAULT_THRESHOLDS, PREPROCESS_VERSION, load_labels_json, resolve_labels_json


@lru_cache(maxsize=32)
def _hash_file(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """Hash large files once per filesystem identity, without loading them whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(path: Path) -> str:
    """Return a stable content hash; represent absent files without breaking settings."""
    if not path.is_file():
        return "missing:" + str(path)
    stat = path.stat()
    return _hash_file(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=8)
def _validate_labels(path: str, digest: str) -> None:
    """Parse once per content digest; 破損 JSON を署名段階で検出し、再タグ不要時の素通りを防ぐ."""
    load_labels_json(Path(path))


def signature_labels_path(settings: TaggerSettings) -> Path:
    """
    明示 labels_json の欠落・破損は必ず ValueError. 隣接ファイルの自動探索は未指定時のみ.
    未指定かつ隣接ファイルも無い場合は "missing:" 署名とし、バックエンド生成時にエラーとなる.
    """
    model = Path(settings.model_path or "__missing_model__").expanduser()
    if settings.labels_json:
        labels = resolve_labels_json(model, settings.labels_json)
    else:
        try:
            labels = resolve_labels_json(model)
        except ValueError:
            return model.parent / "tags.json"
    _validate_labels(str(labels.resolve()), content_digest(labels))
    return labels


def v1_content_identity(model_path: str | Path, labels_path: str | Path) -> dict[str, str]:
    """Digest every file the ONNX session and label map read."""
    model = Path(model_path).expanduser()
    # Both supported distributions either use model.onnx.data or embed weights.
    weights = model.with_name(model.name + ".data")
    return dict(
        model=content_digest(model),
        weights=content_digest(weights) if weights.is_file() else "embedded",
        labels=content_digest(Path(labels_path)),
    )


def settings_content_identity(settings: TaggerSettings) -> dict[str, str]:
    """Content identity of the files the current settings point to."""
    model = Path(settings.model_path or "__missing_model__").expanduser()
    return v1_content_identity(model, signature_labels_path(settings))


def v1_signature(
    settings: TaggerSettings,
    *,
    thresholds: ThresholdMap | None,
    max_tags: MaxTagsMap | None,
    identity: Mapping[str, str] | None = None,
) -> str:
    """
    Include ONNX, external weights, label order, preprocessing and effective options.
    identity を渡すと同じ内容スナップショットから署名を作る（署名と照合用 identity の食い違いを防ぐ）.
    """
    from tagger.base import TagCategory

    threshold_values = settings.thresholds if thresholds is None else {c.name.lower(): v for c, v in thresholds.items()}
    threshold_values = dict(DEFAULT_THRESHOLDS) | dict(threshold_values)
    payload = dict(
        dict(identity) if identity is not None else settings_content_identity(settings),
        preprocess=PREPROCESS_VERSION,
        thresholds={str(k): float(v) for k, v in threshold_values.items()},
        max_tags={TagCategory(k).name.lower(): int(v) for k, v in (max_tags or {}).items()},
    )
    return "pixai-v1-onnx:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_loaded_identity(tagger: object, expected: Mapping[str, str] | None, *, signed: bool) -> None:
    """
    署名付きで DB に書く前の共通契約. tagger がロードした内容と、署名の計算に使った内容が一致しなければ中止する.
    content_identity を持たない tagger（v1 以外・テスト用）は対象外. 署名なしの書き込みは食い違いが生じないため許可.
    """
    loaded = getattr(tagger, "content_identity", None)
    if loaded is None or not signed:
        return
    if expected is None or dict(loaded) != dict(expected):
        raise RuntimeError(
            "PixAI v1 loaded model differs from the files used for the tagger signature; reload the tagger"
        )
