"""View model encapsulating settings-related behaviour for the UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

from PyQt6.QtCore import QObject, pyqtSignal

from core.config.schema import PipelineSettings, TaggerSettings
from db.admin import reset_database as _reset_database
from utils.paths import get_db_path as _get_db_path

logger = logging.getLogger(__name__)


def _default_provider_loader() -> list[str]:
    from tagger import wd14_onnx

    return wd14_onnx.get_available_providers()


class SettingsViewModel(QObject):
    """Coordinate persistence-free settings operations for :class:`ui.settings_tab.SettingsTab`."""

    settings_applied = pyqtSignal(PipelineSettings)
    tagger_environment_checked = pyqtSignal(str)
    database_reset = pyqtSignal(dict)
    database_reset_failed = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        db_path: Path | None = None,
        reset_database: Callable[[Path, bool], dict[str, object]] = _reset_database,
        provider_loader: Callable[[], Iterable[str]] = _default_provider_loader,
    ) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path) if db_path is not None else Path(_get_db_path())
        self._reset_database = reset_database
        self._provider_loader = provider_loader
        self._current_settings = PipelineSettings()

    @property
    def db_path(self) -> Path:
        """Return the active database path."""

        return self._db_path

    @property
    def current_settings(self) -> PipelineSettings:
        """Return the cached settings."""

        return self._current_settings

    def set_current_settings(self, settings: PipelineSettings) -> None:
        """Update the cached settings without emitting signals."""

        self._current_settings = settings

    def build_settings(
        self,
        *,
        roots: Iterable[Path],
        excluded: Iterable[Path],
        batch_size: int,
        tagger_name: str,
        model_path: str | None,
        previous_tagger: TaggerSettings,
    ) -> PipelineSettings:
        """Construct a :class:`PipelineSettings` instance from UI inputs."""

        tagger_settings = TaggerSettings(
            name=tagger_name,
            model_path=model_path,
            tags_csv=previous_tagger.tags_csv if tagger_name.lower() in {"wd14-onnx", "pixai-onnx"} else None,
            thresholds=dict(previous_tagger.thresholds),
        )
        settings = PipelineSettings(
            roots=list(roots),
            excluded=list(excluded),
            batch_size=batch_size,
            tagger=tagger_settings,
        )
        return settings

    def apply_settings(self, settings: PipelineSettings) -> None:
        """Cache and emit the supplied settings."""

        self._current_settings = settings
        self.settings_applied.emit(settings)

    def check_tagger_environment(self) -> str:
        """Inspect available ONNX providers and emit the resulting summary."""

        try:
            providers = list(self._provider_loader())
        except Exception as exc:  # pragma: no cover - defensive logging
            message = str(exc)
            logger.warning("Tagger environment check failed: %s", exc)
        else:
            if providers:
                joined = ", ".join(providers)
            else:
                joined = "<none>"
            message = f"ONNX providers: {joined}"
            logger.info("Available ONNX providers: %s", joined)
        self.tagger_environment_checked.emit(message)
        return message

    def reset_database(self, *, backup: bool) -> dict[str, object]:
        """Reset the database via :func:`db.admin.reset_database`."""

        try:
            result = self._reset_database(self._db_path, backup=backup)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Database reset failed for %s", self._db_path)
            message = str(exc)
            self.database_reset_failed.emit(message)
            raise
        else:
            self.database_reset.emit(result)
            return result


__all__ = ["SettingsViewModel"]
