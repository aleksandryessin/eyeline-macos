#!/usr/bin/env python3
"""Download EyeLine's pinned runtime models with SHA-256 verification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from eyeline.models.downloads import download_runtime_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "EYELINE_MODEL_DIR", Path.home() / ".local" / "share" / "eyeline" / "models"
            )
        ),
        help="artifact directory (default: EYELINE_MODEL_DIR or user data directory)",
    )
    args = parser.parse_args()
    checkpoints, landmarker = download_runtime_models(args.model_dir.expanduser())
    print(f"Gaze checkpoints: {checkpoints}")
    print(f"Face Landmarker: {landmarker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
