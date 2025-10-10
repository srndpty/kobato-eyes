"""Configuration domain primitives for kobato-eyes."""

from __future__ import annotations

from pathlib import Path

from .paths import AppPaths
from .schema import PipelineSettings, TaggerSettings
from .service import SettingsService

_APP_PATHS = AppPaths()
_SERVICE = SettingsService(_APP_PATHS)


def configure(app_paths: AppPaths) -> None:
    """Replace the default :class:`SettingsService` dependencies."""

    global _APP_PATHS, _SERVICE
    _APP_PATHS = app_paths
    _SERVICE = SettingsService(_APP_PATHS)


def config_path() -> Path:
    """Return the path to the configuration file."""

    return _SERVICE.config_path


def load_settings() -> PipelineSettings:
    """Load pipeline settings using the shared service."""

    return _SERVICE.load()


def save_settings(settings: PipelineSettings) -> None:
    """Persist pipeline settings using the shared service."""

    _SERVICE.save(settings)


__all__ = [
    "AppPaths",
    "PipelineSettings",
    "SettingsService",
    "TaggerSettings",
    "config_path",
    "configure",
    "load_settings",
    "save_settings",
]
