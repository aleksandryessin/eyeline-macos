from __future__ import annotations

import argparse
from pathlib import Path


def _shape(value: str) -> tuple[int, int, int, int]:
    try:
        dimensions = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from error
    if len(dimensions) != 4 or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("shape must have four positive dimensions")
    return dimensions  # type: ignore[return-value]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Convert an EyeLine TensorFlow SavedModel to Core ML")
    result.add_argument("saved_model", type=Path)
    result.add_argument("output", type=Path)
    result.add_argument("--input-name", required=True)
    result.add_argument("--input-shape", required=True, type=_shape, metavar="1,H,W,C")
    result.add_argument("--minimum-macos", choices=("13", "14", "15"), default="14")
    return result


def main() -> None:
    arguments = parser().parse_args()
    from .converter import convert_saved_model

    convert_saved_model(
        arguments.saved_model,
        arguments.output,
        input_name=arguments.input_name,
        input_shape=arguments.input_shape,
        minimum_macos=arguments.minimum_macos,
    )


if __name__ == "__main__":
    main()
