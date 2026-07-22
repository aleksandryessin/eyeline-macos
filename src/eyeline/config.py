"""Validated configuration shared across EyeLine entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CameraConfig:
    # None means the physical built-in Mac camera. A numeric index is an
    # explicit advanced override and may refer to an external camera.
    index: int | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    strength: float = 0.55
    model_backend: str = "tensorflow"
    focal_length: float | None = None


@dataclass(frozen=True, slots=True)
class LandmarkConfig:
    backend: str = "auto"
    min_confidence: float = 0.55


@dataclass(frozen=True, slots=True)
class NaturalizerConfig:
    smoothing_alpha: float = 0.35
    fade_in_seconds: float = 0.20
    fade_out_seconds: float = 0.12
    head_turn_soft_limit_degrees: float = 18.0
    head_turn_hard_limit_degrees: float = 32.0
    extreme_gaze_threshold: float = 0.82
    blink_threshold: float = 0.18


@dataclass(frozen=True, slots=True)
class OutputConfig:
    backend: str = "obs"
    device: str = "OBS Virtual Camera"


@dataclass(frozen=True, slots=True)
class EyeLineConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    naturalizer: NaturalizerConfig = field(default_factory=NaturalizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.camera.index is not None and self.camera.index < 0:
            raise ValueError("camera index must be non-negative")
        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if not 0.0 <= self.correction.strength <= 1.0:
            raise ValueError("correction strength must be between 0 and 1")
        if self.correction.focal_length is not None and self.correction.focal_length <= 0.0:
            raise ValueError("correction focal length must be positive")
        if not 0.0 <= self.landmarks.min_confidence <= 1.0:
            raise ValueError("landmark confidence must be between 0 and 1")
        if not 0.0 < self.naturalizer.smoothing_alpha <= 1.0:
            raise ValueError("smoothing alpha must be in (0, 1]")


def load_config(path: str | Path) -> EyeLineConfig:
    """Load YAML into immutable validated config objects."""

    raw: dict[str, Any]
    with Path(path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    return EyeLineConfig(
        camera=CameraConfig(**raw.get("camera", {})),
        correction=CorrectionConfig(**raw.get("correction", {})),
        landmarks=LandmarkConfig(**raw.get("landmarks", {})),
        naturalizer=NaturalizerConfig(**raw.get("naturalizer", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
