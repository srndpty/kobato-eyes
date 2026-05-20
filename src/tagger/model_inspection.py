"""Lightweight inspection helpers for configured tagger models."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from tagger.labels_util import TagMeta, discover_labels_csv, load_selected_tags
from tagger.onnx_backend import CPU_PROVIDER, validate_label_count

_PIXAI_STRONG_OUTPUT_NAMES: frozenset[str] = frozenset({"prediction"})
_PIXAI_WEAK_OUTPUT_NAMES: frozenset[str] = frozenset({"logits"})
_PIXAI_EXPECTED_LABEL_COUNT = 13461

_CATEGORY_LABELS: dict[int, str] = {
    0: "general",
    1: "character",
    2: "rating",
    3: "copyright",
    4: "artist",
    5: "meta",
}


class _OnnxOutput(Protocol):
    """Subset of ONNX Runtime output metadata used by inspection."""

    name: str
    shape: list[Any]


class _OnnxSession(Protocol):
    """Subset of ONNX Runtime session used by inspection."""

    def get_outputs(self) -> list[_OnnxOutput]:
        """Return model output metadata."""

    def get_modelmeta(self) -> Any:
        """Return optional model metadata."""


SessionFactory = Callable[[Path, list[str] | None], _OnnxSession]


@dataclass(frozen=True)
class ModelInspection:
    """User-facing summary of a tagger model and its label CSV."""

    ok: bool
    tagger_name: str
    model_path: Path | None = None
    labels_csv: Path | None = None
    model_name: str | None = None
    label_total: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    output_dim: int | None = None
    output_name: str | None = None
    provider: str = "wd14"
    providers: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def _default_session_factory(model_path: Path, providers: list[str] | None) -> _OnnxSession:
    """Create an ONNX Runtime session for model inspection."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to validate the model file. Try: pip install onnxruntime-gpu "
            "(or onnxruntime for CPU)"
        ) from exc
    return ort.InferenceSession(str(model_path), providers=providers)


def _normalise_providers(providers: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(provider) for provider in providers if str(provider))


def _inspection_providers(available: tuple[str, ...]) -> list[str] | None:
    if CPU_PROVIDER in available:
        return [CPU_PROVIDER]
    if available:
        return [available[0]]
    return None


def _category_counts(labels: Iterable[TagMeta]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        key = _CATEGORY_LABELS.get(int(label.category), f"category_{int(label.category)}")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _output_dim(output: _OnnxOutput) -> int | None:
    shape = list(getattr(output, "shape", []) or [])
    if not shape:
        return None
    last = shape[-1]
    return int(last) if isinstance(last, int) else None


def _looks_like_pixai_output(output_name: str, output_dim: int | None) -> bool:
    if output_name in _PIXAI_STRONG_OUTPUT_NAMES:
        return output_dim == _PIXAI_EXPECTED_LABEL_COUNT
    if output_name in _PIXAI_WEAK_OUTPUT_NAMES:
        return output_dim == _PIXAI_EXPECTED_LABEL_COUNT
    return False


def _matches_label_count(output_dim: int | None, label_count: int | None) -> bool:
    return output_dim is not None and label_count is not None and output_dim == label_count


def _prediction_output(session: _OnnxSession, *, label_count: int | None) -> tuple[str | None, int | None, str]:
    outputs = session.get_outputs()
    if not outputs:
        raise RuntimeError("ONNX model has no outputs")

    by_name = {str(getattr(output, "name", "")): output for output in outputs}
    for name in sorted(_PIXAI_STRONG_OUTPUT_NAMES):
        candidate = by_name.get(name)
        if candidate is not None:
            output_dim = _output_dim(candidate)
            if _looks_like_pixai_output(name, output_dim) or _matches_label_count(output_dim, label_count):
                return name, output_dim, "pixai"

    for name in sorted(_PIXAI_WEAK_OUTPUT_NAMES):
        candidate = by_name.get(name)
        if candidate is not None:
            output_dim = _output_dim(candidate)
            provider = "pixai" if _looks_like_pixai_output(name, output_dim) else "wd14"
            return name, output_dim, provider

    if label_count is not None:
        for output in outputs:
            output_dim = _output_dim(output)
            if output_dim == label_count:
                return str(getattr(output, "name", "") or ""), output_dim, "wd14"

    selected = outputs[0]
    return str(getattr(selected, "name", "") or ""), _output_dim(selected), "wd14"


def _metadata_from_session(session: _OnnxSession) -> dict[str, str]:
    try:
        meta = session.get_modelmeta()
    except Exception:
        return {}

    values: dict[str, str] = {}
    for attr in ("graph_name", "producer_name", "domain", "description", "version"):
        value = getattr(meta, attr, None)
        if value not in (None, ""):
            values[attr] = str(value)
    custom = getattr(meta, "custom_metadata_map", None) or {}
    for key, value in custom.items():
        if value not in (None, ""):
            values[f"custom.{key}"] = str(value)
    return values


@lru_cache(maxsize=16)
def _detect_provider_from_model_outputs_cached(
    path_str: str,
    mtime_ns: int,
    size: int,
    label_count: int | None,
) -> str | None:
    del mtime_ns, size
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    available = set(ort.get_available_providers())
    providers = [CPU_PROVIDER] if CPU_PROVIDER in available else None
    try:
        session = ort.InferenceSession(path_str, providers=providers)
        output_dims = {str(output.name): _output_dim(output) for output in session.get_outputs()}
    except Exception:
        return None
    for name, output_dim in output_dims.items():
        if _looks_like_pixai_output(name, output_dim) or (
            name in _PIXAI_STRONG_OUTPUT_NAMES and _matches_label_count(output_dim, label_count)
        ):
            return "pixai"
    return None


def detect_provider_from_model_outputs(model_path: str | Path | None, *, label_count: int | None = None) -> str | None:
    """Return ``pixai`` when ONNX outputs expose PixAI prediction tensors."""

    if not model_path:
        return None
    path = Path(model_path).expanduser()
    if not path.is_file():
        return None
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError:
        return None
    return _detect_provider_from_model_outputs_cached(
        str(resolved),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        label_count,
    )


def inspect_model(
    *,
    tagger_name: str,
    model_path: str | Path | None,
    tags_csv: str | Path | None,
    provider_loader: Callable[[], Iterable[str]],
    session_factory: SessionFactory = _default_session_factory,
) -> ModelInspection:
    """Inspect a configured tagger model and labels CSV without running inference."""

    lowered = tagger_name.lower()
    if lowered != "wd14-onnx":
        return ModelInspection(ok=True, tagger_name=tagger_name, model_name=tagger_name)
    if not model_path:
        return ModelInspection(ok=False, tagger_name=tagger_name, errors=("WD14 model path is not configured.",))

    path = Path(model_path).expanduser()
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return ModelInspection(
            ok=False,
            tagger_name=tagger_name,
            model_path=path,
            model_name=path.stem,
            errors=(f"WD14 model not found at {path}",),
        )

    labels_csv = discover_labels_csv(path, tags_csv)
    labels: list[TagMeta] = []
    if labels_csv is None:
        errors.append("selected_tags.csv was not found next to the model.")
    else:
        try:
            labels = load_selected_tags(labels_csv)
        except Exception as exc:
            errors.append(f"Failed to parse labels CSV: {exc}")
        if labels_csv is not None and not labels:
            errors.append("No tags were parsed from labels CSV.")

    try:
        available = _normalise_providers(provider_loader())
    except Exception as exc:
        available = ()
        warnings.append(f"Failed to inspect ONNX providers: {exc}")
    session: _OnnxSession | None = None
    output_dim: int | None = None
    output_name: str | None = None
    provider = "wd14"
    metadata: dict[str, str] = {}
    try:
        session = session_factory(path, _inspection_providers(available))
    except Exception as exc:
        errors.append(f"Failed to load ONNX model: {exc}")
    if session is not None:
        try:
            output_name, output_dim, provider = _prediction_output(
                session,
                label_count=len(labels) if labels else None,
            )
            if output_dim is None:
                warnings.append("Model output dimension is dynamic; label count could not be compared.")
            elif labels:
                validate_label_count(output_dim, len(labels), backend_name="PixAI" if provider == "pixai" else "WD14")
        except Exception as exc:
            errors.append(str(exc))
        metadata = _metadata_from_session(session)

    model_name = metadata.get("custom.model_name") or metadata.get("graph_name") or path.stem
    return ModelInspection(
        ok=not errors,
        tagger_name=tagger_name,
        model_path=path,
        labels_csv=labels_csv,
        model_name=model_name,
        label_total=len(labels),
        category_counts=_category_counts(labels),
        output_dim=output_dim,
        output_name=output_name,
        provider=provider,
        providers=available,
        metadata=metadata,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def format_inspection(inspection: ModelInspection) -> str:
    """Format inspection details for the Settings tab diagnostics panel."""

    if inspection.tagger_name.lower() != "wd14-onnx":
        return f"Tagger: {inspection.tagger_name}\nModel validation is not required for this tagger."

    status = "OK" if inspection.ok else "Error"
    lines = [f"Model status: {status}"]
    if inspection.model_name:
        lines.append(f"Model: {inspection.model_name}")
    if inspection.model_path is not None:
        lines.append(f"Model path: {inspection.model_path}")
    if inspection.labels_csv is not None:
        lines.append(f"Tags CSV: {inspection.labels_csv}")
    if inspection.label_total:
        category_text = ", ".join(f"{name}={count}" for name, count in inspection.category_counts.items())
        lines.append(f"Tags: {inspection.label_total} total ({category_text})")
    if inspection.output_dim is not None:
        if inspection.output_name:
            lines.append(f"Prediction output: {inspection.output_name}")
        lines.append(f"Output dimension: {inspection.output_dim}")
    if inspection.provider:
        lines.append(f"Detected tagger backend: {inspection.provider}")
    if inspection.providers:
        lines.append(f"ONNX providers: {', '.join(inspection.providers)}")
    for warning in inspection.warnings:
        lines.append(f"Warning: {warning}")
    for error in inspection.errors:
        lines.append(f"Error: {error}")
    return "\n".join(lines)


__all__ = ["ModelInspection", "detect_provider_from_model_outputs", "format_inspection", "inspect_model"]
