"""Image input/output utilities built atop Pillow."""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Optional, TYPE_CHECKING, cast

from PIL import Image, ImageFile, UnidentifiedImageError
from PIL.Image import DecompressionBombError

if TYPE_CHECKING:
    from PyQt6.QtCore import Qt as QtType
    from PyQt6.QtGui import QPixmap as QPixmapType
else:  # pragma: no cover - imported lazily at runtime
    QtType = Any  # type: ignore[assignment]
    QPixmapType = Any  # type: ignore[assignment]

try:  # pragma: no cover - handled at runtime when PyQt6 is missing
    from PyQt6.QtCore import Qt as _Qt
    from PyQt6.QtGui import QPixmap as _QPixmap
except ImportError:  # pragma: no cover - simplifies headless testing environments
    _Qt = None
    _QPixmap = None

Qt: QtType | None = cast("QtType | None", _Qt)
QPixmap: QPixmapType | None = cast("QPixmapType | None", _QPixmap)

from utils.paths import ensure_dirs, get_cache_dir

DEFAULT_THUMBNAIL_SIZE = (320, 320)
_THUMB_CACHE_LIMIT = 256
_THUMB_CACHE: "OrderedDict[str, QPixmapType]" = OrderedDict()
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
    bomb_pixel_cap: Optional[int] = DEFAULT_BOMB_CAP,
    hard_skip_pixels: Optional[int] = None,  # ← 追加: これ超は開かない
    rgb: bool = True,
    skip_on_bomb: bool = False,
):
    p = str(source)
    old_cap = Image.MAX_IMAGE_PIXELS
    if bomb_pixel_cap is not None:
        Image.MAX_IMAGE_PIXELS = int(bomb_pixel_cap)

    try:
        # まずはヘッダだけでサイズを取る（ここは低コスト）
        img: Image.Image = Image.open(p)  # デコード前
        w, h = img.size
        px = (w or 0) * (h or 0)

        # 超巨大は最初からスキップ（ここが肝）
        if hard_skip_pixels is not None and px > hard_skip_pixels:
            logger.warning("Skip very large image (header %dx%d ~%d px): %s", w, h, px, p)
            try:
                img.close()
            except Exception:
                pass
            return None

        # JPEG 等は draft で縮小デコードを促す
        try:
            img.draft("RGB", (max_side, max_side))
        except Exception:
            pass

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img.load()  # ここで実デコード
        except DecompressionBombError as e:
            logger.warning("Huge image detected (bomb): %s (%s)", p, e)
            if skip_on_bomb:
                try:
                    img.close()
                except Exception:
                    pass
                return None
            # 通す場合は制限を外して開き直し
            try:
                img.close()
            except Exception:
                pass
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(p)
            try:
                img.draft("RGB", (max_side, max_side))
            except Exception:
                pass
            img.load()
        except MemoryError:
            # ここで落ちていた。スキップに切り替え
            logger.error("MemoryError while decoding (header %dx%d ~%d px): %s", w, h, px, p)
            try:
                img.close()
            except Exception:
                pass
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


if TYPE_CHECKING:
    from PIL.Image import Resampling
else:  # pragma: no cover - used only for typing
    Resampling = Any  # type: ignore[assignment]


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    *,
    resample: Resampling | None = None,
) -> Image.Image:
    """Return a resized copy of ``image`` constrained to ``size`` bounds."""
    if resample is None:
        resampling = getattr(Image, "Resampling", None)
        if resampling is not None:  # Pillow >= 9.1
            resample_value = cast(
                "Resampling",
                getattr(resampling, "LANCZOS", getattr(Image, "BICUBIC", Image.BILINEAR)),
            )
        else:
            resample_value = cast(
                "Resampling",
                getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", Image.BILINEAR)),
            )
    else:
        resample_value = resample
    copy = image.copy()
    copy.thumbnail(size, resample_value)
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

    image = safe_load_image(source)
    # image = safe_load_image(source, mode)
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
) -> QPixmapType:
    """Return a cached QPixmap thumbnail for ``source_path`` resized to fit within ``width``×``height``."""
    if QPixmap is None or Qt is None:
        raise RuntimeError(
            "QPixmap is unavailable in this environment. Install PyQt6 with OpenGL support or unset KOE_HEADLESS."
        )
    assert QPixmap is not None  # narrow type for type-checkers
    assert Qt is not None
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
