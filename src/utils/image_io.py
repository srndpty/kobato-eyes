"""Image input/output utilities built atop Pillow."""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from threading import Lock

from PIL import Image, ImageFile, UnidentifiedImageError
from PIL.Image import DecompressionBombError

try:  # pragma: no cover - handled at runtime when PyQt6 is missing
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
except ImportError:  # pragma: no cover - simplifies headless testing environments
    Qt = None  # type: ignore[assignment]
    QPixmap = None  # type: ignore[assignment]

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


logger = logging.getLogger(__name__)

# 既定の上限（Pillowデフォルトは ~1.79e8 px）
DEFAULT_BOMB_CAP = 350_000_000  # 3.5e8 px まで許容（開いたら即縮小）
DEFAULT_HARD_SKIP = 220_000_000  # これ超は埋め込みではスキップ
DEFAULT_MAX_SIDE = 4096  # ここまでに縮小して返す


def safe_load_image(
    source: str | Path,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    bomb_pixel_cap: int | None = DEFAULT_BOMB_CAP,
    hard_skip_pixels: int | None = None,  # ← 追加: これ超は開かない
    rgb: bool = True,
    skip_on_bomb: bool = False,
) -> Image.Image | None:
    """Load an image defensively and return ``None`` for unsafe or unreadable files."""
    p = str(source)
    old_cap = Image.MAX_IMAGE_PIXELS
    if bomb_pixel_cap is not None:
        Image.MAX_IMAGE_PIXELS = int(bomb_pixel_cap)

    try:
        # まずはヘッダだけでサイズを取る（ここは低コスト）
        img = Image.open(p)  # デコード前
        w, h = img.size
        px = (w or 0) * (h or 0)

        # 超巨大は最初からスキップ（ここが肝）
        if hard_skip_pixels is not None and px > hard_skip_pixels:
            logger.warning("Skip very large image (header %dx%d ~%d px): %s", w, h, px, p)
            with suppress(Exception):
                img.close()
            return None

        # JPEG 等は draft で縮小デコードを促す
        with suppress(Exception):
            img.draft("RGB", (max_side, max_side))

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img.load()  # ここで実デコード
        except DecompressionBombError as e:
            logger.warning("Huge image detected (bomb): %s (%s)", p, e)
            if skip_on_bomb:
                with suppress(Exception):
                    img.close()
                return None
            # 通す場合は制限を外して開き直し
            with suppress(Exception):
                img.close()
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(p)
            with suppress(Exception):
                img.draft("RGB", (max_side, max_side))
            img.load()
        except MemoryError:
            # ここで落ちていた。スキップに切り替え
            logger.error("MemoryError while decoding (header %dx%d ~%d px): %s", w, h, px, p)
            with suppress(Exception):
                img.close()
            return None

        # ここまで来たら即縮小して RAM を抑える
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        if rgb and img.mode != "RGB":
            img = img.convert("RGB")
        return img

    except (UnidentifiedImageError, OSError) as e:
        logger.warning("safe_load_image failed for %s: %s", p, e)
        return None
    finally:
        Image.MAX_IMAGE_PIXELS = old_cap


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


def _cached_thumbnail_is_readable(path: Path) -> bool:
    """Return whether an existing thumbnail cache entry can still be decoded."""
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        logger.warning("Discarding corrupt thumbnail cache entry: %s", path)
        with suppress(OSError):
            path.unlink()
        return False
    return True


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
        if _cached_thumbnail_is_readable(thumb_path):
            return thumb_path

    image = safe_load_image(source)
    # image = safe_load_image(source, mode)
    if image is None:
        return None

    tmp_path = thumb_path.with_suffix(thumb_path.suffix + ".tmp")
    try:
        thumbnail = resize_image(image, size)
        thumbnail.save(tmp_path, format=format)
        os.replace(tmp_path, thumb_path)
        return thumb_path
    except OSError as exc:
        logger.warning("Failed to write thumbnail cache for %s: %s", source, exc)
        with suppress(OSError):
            tmp_path.unlink()
        return None
    finally:
        with suppress(Exception):
            image.close()


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
