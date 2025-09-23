"""Image input/output utilities built atop Pillow."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from utils.fs import to_system_path

DEFAULT_THUMBNAIL_SIZE = (320, 320)


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


def _thumbnail_cache_key(
    path: Path, size: tuple[int, int], mode: str | None, fmt: str
) -> str:
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


__all__ = [
    "safe_load_image",
    "resize_image",
    "generate_thumbnail",
    "DEFAULT_THUMBNAIL_SIZE",
]
