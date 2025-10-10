"""Services for loading and persisting pipeline configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .paths import AppPaths
from .schema import PipelineSettings

logger = logging.getLogger(__name__)


class SettingsService:
    """Load, validate and persist :class:`PipelineSettings`."""

    def __init__(self, app_paths: AppPaths, *, filename: str = "config.yaml") -> None:
        self._app_paths = app_paths
        self._filename = filename

    @property
    def config_path(self) -> Path:
        """Path to the configuration file."""

        path = self._app_paths.config_path(self._filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load(self) -> PipelineSettings:
        """Load the configuration from disk with graceful fallbacks."""

        path = self.config_path
        if not path.exists():
            return PipelineSettings()

        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Unable to read settings from %s: %s", path, exc)
            return PipelineSettings()

        try:
            raw_data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            logger.warning("Invalid YAML in %s: %s", path, exc)
            return PipelineSettings()

        settings = PipelineSettings.from_mapping(raw_data)

        default_index_dir = str(self._app_paths.index_dir())
        if settings.index_dir == default_index_dir:
            settings.index_dir = None

        return settings

    def save(self, settings: PipelineSettings) -> None:
        """Persist the configuration to disk."""

        path = self.config_path
        payload = settings.to_mapping(default_index_dir=str(self._app_paths.index_dir()))
        try:
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to write settings to %s: %s", path, exc)
            raise


__all__ = ["SettingsService"]
