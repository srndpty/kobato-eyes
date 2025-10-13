"""Tests for WD14 tagger input layout utilities."""

from __future__ import annotations

import numpy as np

from tagger.wd14_onnx import WD14Tagger


def test_infer_input_layout_nchw() -> None:
    shape = (None, 3, 448, 448)
    layout = WD14Tagger._infer_input_layout(shape)
    assert layout == "NCHW"
    height, width = WD14Tagger._infer_spatial_dims(shape, layout)
    assert height == 448
    assert width == 448


def test_infer_input_layout_nhwc() -> None:
    shape = (None, 448, 448, 3)
    layout = WD14Tagger._infer_input_layout(shape)
    assert layout == "NHWC"
    height, width = WD14Tagger._infer_spatial_dims(shape, layout)
    assert height == 448
    assert width == 448


def test_format_for_session_input_transposes_to_nchw() -> None:
    tagger = object.__new__(WD14Tagger)
    tagger._input_layout = "NCHW"  # type: ignore[attr-defined]
    batch = np.zeros((2, 448, 448, 3), dtype=np.float32)
    formatted = WD14Tagger._format_for_session_input(tagger, batch)
    assert formatted.shape == (2, 3, 448, 448)


def test_format_for_session_input_keeps_nhwc() -> None:
    tagger = object.__new__(WD14Tagger)
    tagger._input_layout = "NHWC"  # type: ignore[attr-defined]
    batch = np.zeros((2, 448, 448, 3), dtype=np.float32)
    formatted = WD14Tagger._format_for_session_input(tagger, batch)
    assert formatted.shape == (2, 448, 448, 3)
