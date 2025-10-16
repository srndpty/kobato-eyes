"""Application icon helpers for kobato-eyes."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PyQt6.QtGui import QIcon

_ICON_DIR: Final[Path] = Path(__file__).resolve().parent / "assets" / "icons"
_RIGHT_EYE_FILE: Final[str] = "kobato_eye_right.png"
_LEFT_EYE_FILE: Final[str] = "kobato_eye_left.png"
_RIGHT_EYE_EXE_FILE: Final[str] = "kobato_eye_right.ico"


def _resolve_resource(name: str) -> Path:
    """Return the absolute path to an icon resource."""

    path = _ICON_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


class EyeIconProvider:
    """Provide access to the application's themed eye icons."""

    def __init__(self) -> None:
        self._right_eye_icon = QIcon(str(_resolve_resource(_RIGHT_EYE_FILE)))
        self._left_eye_icon = QIcon(str(_resolve_resource(_LEFT_EYE_FILE)))

    @property
    def right_eye(self) -> QIcon:
        """Return the icon representing the right (red) eye."""

        return self._right_eye_icon

    @property
    def left_eye(self) -> QIcon:
        """Return the icon representing the left (blue) eye."""

        return self._left_eye_icon

    @property
    def executable_icon_path(self) -> Path:
        """Return the path to the icon file suitable for Windows executables."""

        return _resolve_resource(_RIGHT_EYE_EXE_FILE)
