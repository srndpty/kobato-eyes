"""Tests for tagger model inspection helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tagger.model_inspection import format_inspection, inspect_model


class _Session:
    """Minimal ONNX Runtime session stub for inspection tests."""

    def __init__(self, outputs: list[SimpleNamespace]) -> None:
        self._outputs = outputs

    def get_outputs(self) -> list[SimpleNamespace]:
        return self._outputs

    def get_modelmeta(self) -> SimpleNamespace:
        return SimpleNamespace(
            graph_name="wd14-test",
            producer_name="unit-test",
            domain="",
            description="",
            version="",
            custom_metadata_map={"model_name": "WD14 Test Model"},
        )


def _session_factory(output_dim: int):
    def create_session(_model_path: Path, _providers: list[str] | None) -> _Session:
        return _Session([SimpleNamespace(name="tags", shape=[None, output_dim])])

    return create_session


def _multi_output_session_factory() -> _Session:
    return _Session(
        [
            SimpleNamespace(name="embedding", shape=["batch_size", 1024]),
            SimpleNamespace(name="logits", shape=["batch_size", 13461]),
            SimpleNamespace(name="prediction", shape=["batch_size", 13461]),
        ]
    )


def test_inspect_model_reports_labels_metadata_and_providers(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "selected_tags.csv"
    model_path.write_bytes(b"onnx")
    labels_path.write_text(
        "tag_id,name,category,count\n1,1girl,0,100\n2,kobato,1,50\n3,series,3,20\n",
        encoding="utf-8",
    )

    inspection = inspect_model(
        tagger_name="wd14-onnx",
        model_path=model_path,
        tags_csv=None,
        provider_loader=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=_session_factory(3),
    )

    assert inspection.ok is True
    assert inspection.model_name == "WD14 Test Model"
    assert inspection.labels_csv == labels_path
    assert inspection.label_total == 3
    assert inspection.category_counts == {"character": 1, "copyright": 1, "general": 1}
    assert inspection.output_dim == 3
    assert inspection.providers == ("CUDAExecutionProvider", "CPUExecutionProvider")

    message = format_inspection(inspection)
    assert "Model status: OK" in message
    assert "Tags: 3 total" in message


def test_inspect_model_reports_output_label_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "selected_tags.csv"
    model_path.write_bytes(b"onnx")
    labels_path.write_text("one\n", encoding="utf-8")

    inspection = inspect_model(
        tagger_name="wd14-onnx",
        model_path=model_path,
        tags_csv=labels_path,
        provider_loader=lambda: ["CPUExecutionProvider"],
        session_factory=_session_factory(2),
    )

    assert inspection.ok is False
    assert inspection.errors == ("WD14: model output dimension 2 does not match label count 1",)
    assert "Model status: Error" in format_inspection(inspection)


def test_inspect_model_reports_missing_csv_without_loading_session(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    inspection = inspect_model(
        tagger_name="wd14-onnx",
        model_path=model_path,
        tags_csv=None,
        provider_loader=lambda: ["CPUExecutionProvider"],
        session_factory=_session_factory(1),
    )

    assert inspection.ok is False
    assert "selected_tags.csv was not found" in inspection.errors[0]
    assert inspection.output_dim == 1


def test_inspect_model_selects_pixai_prediction_output(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "selected_tags.csv"
    model_path.write_bytes(b"onnx")
    labels_path.write_text("\n".join(f"{index},tag_{index},0,1" for index in range(13461)), encoding="utf-8")

    inspection = inspect_model(
        tagger_name="wd14-onnx",
        model_path=model_path,
        tags_csv=labels_path,
        provider_loader=lambda: ["CPUExecutionProvider"],
        session_factory=lambda _model_path, _providers: _multi_output_session_factory(),
    )

    assert inspection.ok is True
    assert inspection.provider == "pixai"
    assert inspection.output_name == "prediction"
    assert inspection.output_dim == 13461
    assert "Detected provider: pixai" in format_inspection(inspection)
