"""Core ML conversion entry points.

This environment must stay separate from the Python 3.12 OBS runtime. It loads
an exported TensorFlow SavedModel, not the live compat.v1 runtime graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import coremltools as ct
import tensorflow as tf


def convert_saved_model(
    saved_model: Path,
    output: Path,
    *,
    input_name: str,
    input_shape: Sequence[int],
    minimum_macos: str = "14",
) -> None:
    """Convert an exported SavedModel to an ML Program package.

    `input_shape` includes the batch dimension. The function intentionally does
    not guess checkpoint tensor names; inspect and export the upstream
    compat.v1 graph explicitly before conversion.
    """

    if not saved_model.exists():
        raise FileNotFoundError(saved_model)
    if len(input_shape) != 4 or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape must contain four positive dimensions")

    # Loading first gives a clearer error for malformed/incompatible exports.
    tf.saved_model.load(str(saved_model))

    targets = {
        "13": ct.target.macOS13,
        "14": ct.target.macOS14,
        "15": ct.target.macOS15,
    }
    try:
        target = targets[minimum_macos]
    except KeyError as error:
        raise ValueError("minimum_macos must be one of: 13, 14, 15") from error

    model = ct.convert(
        str(saved_model),
        source="tensorflow",
        convert_to="mlprogram",
        minimum_deployment_target=target,
        inputs=[ct.TensorType(name=input_name, shape=tuple(input_shape))],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
