"""Image input/output utilities built atop Pillow."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from PIL import Image, UnidentifiedImageError

try:  # pragma: no cover - handled at runtime when PyQt6 is missing
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
except ImportError:  # pragma: no cover - simplifies headless testing environments
    Qt = None  # type: ignore[assignment]
    QPixmap = None  # type: ignore[assignment]

from utils.fs import to_system_path
from utils.paths import ensure_dirs, get_cache_dir

DEFAULT_THUMBNAIL_SIZE = (320, 320)
_THUMB_CACHE_LIMIT = 256
_THUMB_CACHE: "OrderedDict[str, QPixmap]" = OrderedDict()
_THUMB_CACHE_LOCK = Lock()


def _thumb_cache_dir() -> Path:
    ensure_dirs()
    cache_dir = get_cache_dir() / "thumbs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def safe_load_image(path: str | Path, mode: str | None = "RGB") -> Image.Image | None:
    """Load an image from disk, returning a copy or ``None`` if decoding fails."""
    source = Path(path)
    try:
        with Image.open(to_system_path(source)) as image:
            image.load()
            if mode is None:
                return image.copy()
            return image.convert(mode).copy()
    except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError):
        return None


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    *,
    resample: int | None = None,
) -> Image.Image:
    """Return a resized copy of ``image`` constrained to ``size`` bounds."""
    if resample is None:
        resampling = getattr(Image, "Resampling", None)
        if resampling is not None:  # Pillow >= 9.1
            resample = resampling.LANCZOS  # type: ignore[assignment]
        else:
            resample = Image.LANCZOS
    copy = image.copy()
    copy.thumbnail(size, resample)
    return copy


def _thumbnail_cache_key(path: Path, size: tuple[int, int], mode: str | None, fmt: str) -> str:
    stat_result = path.stat()
    payload = f"{path.resolve()}|{stat_result.st_size}|{stat_result.st_mtime_ns}|{size}|{mode}|{fmt.lower()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def generate_thumbnail(
    source_path: str | Path,
    cache_dir: str | Path,
    *,
    size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
    mode: str | None = "RGB",
    format: str = "JPEG",
) -> Path | None:
    """Create or retrieve a cached thumbnail for ``source_path``."""
    source = Path(source_path)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        return None

    key = _thumbnail_cache_key(source, size, mode, format)
    thumb_path = cache_root / f"{key}.{format.lower()}"
    if thumb_path.exists():
        return thumb_path

    image = safe_load_image(source, mode)
    if image is None:
        return None

    thumbnail = resize_image(image, size)

    tmp_path = thumb_path.with_suffix(thumb_path.suffix + ".tmp")
    thumbnail.save(tmp_path, format=format)
    os.replace(tmp_path, thumb_path)
    return thumb_path


def get_thumbnail(
    source_path: str | Path,
    width: int = 128,
    height: int = 128,
) -> QPixmap:
    """Return a cached QPixmap thumbnail for ``source_path`` resized to fit within ``width``×``height``."""
    if QPixmap is None or Qt is None:
        raise RuntimeError(
            "QPixmap is unavailable in this environment. Install PyQt6 with OpenGL support or unset KOE_HEADLESS."
        )
    source = Path(source_path)
    key = f"{source.resolve()}::{width}x{height}"

    with _THUMB_CACHE_LOCK:
        cached = _THUMB_CACHE.get(key)
        if cached is not None:
            _THUMB_CACHE.move_to_end(key)
            return QPixmap(cached)

    cache_dir = _thumb_cache_dir()
    thumb_path = generate_thumbnail(
        source,
        cache_dir,
        size=(max(width, 1), max(height, 1)),
        format="WEBP",
    )

    if thumb_path is None or not thumb_path.exists():
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
    else:
        pixmap = QPixmap(str(thumb_path))
        if pixmap.isNull():
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.GlobalColor.transparent)
        else:
            pixmap = pixmap.scaled(
                width,
                height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

    with _THUMB_CACHE_LOCK:
        _THUMB_CACHE[key] = pixmap
        if len(_THUMB_CACHE) > _THUMB_CACHE_LIMIT:
            _THUMB_CACHE.popitem(last=False)

    return QPixmap(pixmap)


__all__ = [
    "safe_load_image",
    "resize_image",
    "generate_thumbnail",
    "get_thumbnail",
    "DEFAULT_THUMBNAIL_SIZE",
]
