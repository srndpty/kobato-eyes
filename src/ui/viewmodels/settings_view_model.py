"""View model encapsulating settings-related behaviour for the UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol, cast

from PyQt6.QtCore import QObject, pyqtSignal

from core.config.schema import PipelineSettings, TaggerSettings
from db.admin import reset_database as _reset_database
from tagger.model_inspection import ModelInspection, format_inspection, inspect_model
from utils.paths import get_db_path as _get_db_path

logger = logging.getLogger(__name__)

TaggerDevice = Literal["auto", "tensorrt", "cuda", "cpu"]


class _ResetDatabase(Protocol):
    """Callable shape used by :class:`SettingsViewModel` for DB reset."""

    def __call__(self, db_path: str | Path, *, backup: bool) -> dict[str, object]:
        """Reset the database and return reset metadata."""


def _default_provider_loader() -> list[str]:
    from tagger import wd14_onnx

    return wd14_onnx.get_available_providers()


def _normalise_tagger_device(value: str | None) -> TaggerDevice:
    """Return a supported tagger execution-device setting."""

    normalized = str(value or "auto").strip().lower()
    if normalized in {"auto", "tensorrt", "cuda", "cpu"}:
        return cast(TaggerDevice, normalized)
    return "auto"


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
        reset_database: _ResetDatabase = _reset_database,
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

    @property
    def provider_loader(self) -> Callable[[], Iterable[str]]:
        """Return the provider loader used by tagger diagnostics."""

        return self._provider_loader

    def build_settings(
        self,
        *,
        roots: Iterable[Path],
        excluded: Iterable[Path],
        batch_size: int,
        tagger_name: str,
        model_path: str | None,
        previous_tagger: TaggerSettings,
        device: str = "auto",
    ) -> PipelineSettings:
        """Construct a :class:`PipelineSettings` instance from UI inputs."""

        tagger_settings = TaggerSettings(
            name=tagger_name,
            model_path=model_path,
            tags_csv=previous_tagger.tags_csv if tagger_name.lower() == "wd14-onnx" else None,
            provider=previous_tagger.provider,
            device=_normalise_tagger_device(device),
            thresholds=dict(previous_tagger.thresholds),
        )
        settings = PipelineSettings(
            roots=[str(path) for path in roots],
            excluded=[str(path) for path in excluded],
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

    def inspect_tagger_model(
        self,
        *,
        tagger_name: str,
        model_path: str | None,
        tags_csv: str | None = None,
    ) -> ModelInspection:
        """Inspect the configured tagger model and labels CSV."""

        return inspect_model(
            tagger_name=tagger_name,
            model_path=model_path,
            tags_csv=tags_csv,
            provider_loader=self._provider_loader,
        )

    def format_model_inspection(self, inspection: ModelInspection) -> str:
        """Return Settings-tab text for a model inspection result."""

        return format_inspection(inspection)

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
