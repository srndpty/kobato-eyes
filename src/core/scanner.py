"""Filesystem scanning utilities for kobato-eyes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    # ".gif",
    ".bmp",
    ".webp",
    ".tiff",
}


def _resolve(p: Path) -> Path:
    p = Path(p).expanduser()
    try:
        return p.resolve(strict=False)
    except OSError:
        return p.absolute()


def _normalise_exts(extensions: Iterable[str] | None) -> set[str]:
    base = extensions or DEFAULT_EXTENSIONS
    out: set[str] = set()
    for e in base:
        s = str(e).lower()
        if not s.startswith("."):
            s = "." + s
        out.add(s)
    return out


def _is_under(path: Path, parent: Path) -> bool:
    # Python 3.10 互換：relative_to で判定
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_images(
    roots: Sequence[Path | str],
    *,
    excluded: Sequence[Path | str] | None = None,
    extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    """
    画像ファイルを再帰列挙する：
      - パス区切りは pathlib 基準
      - 拡張子は大小無視 & 先頭ドットの有無どちらでもOK
      - excluded 配下は除外
      - 「.*」で始まるドット隠し（ファイル/ディレクトリ）は除外（Windowsの属性までは見ない）
    """
    exts = _normalise_exts(extensions)
    exc = [_resolve(Path(p)) for p in (excluded or [])]

    for r in roots:
        root = _resolve(Path(r))
        if not root.exists():
            continue

        for p in root.rglob("*"):
            # ファイルのみ
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue

            # 拡張子フィルタ
            if p.suffix.lower() not in exts:
                continue

            # 除外パス配下の除外
            if any(_is_under(p, e) for e in exc):
                continue

            # ドット隠し（root からの相対で判定）
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p  # 念のため
            if any(part.startswith(".") for part in rel.parts):
                continue

            yield p


__all__ = ["iter_images"]
