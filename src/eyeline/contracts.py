"""Stable boundaries shared by the engine, OBS runner, and native experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

UInt8Frame = NDArray[np.uint8]
FloatPoints = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FaceGeometry:
    """Normalized face geometry returned by every landmark implementation."""

    landmarks: FloatPoints
    left_eye: FloatPoints
    right_eye: FloatPoints
    confidence: float
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    left_eye_openness: float = 1.0
    right_eye_openness: float = 1.0
    gaze_extremity: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """A BGR frame and diagnostics; frame_bgr must always be safe to publish."""

    frame_bgr: UInt8Frame
    timestamp: float
    face_detected: bool
    correction_applied: bool
    effective_strength: float
    processing_ms: float
    reason: str | None = None


class LandmarkBackend(Protocol):
    """Landmark implementations receive RGB, never OpenCV BGR."""

    def detect(self, rgb: UInt8Frame) -> FaceGeometry | None: ...


class FrameProcessor(Protocol):
    """Processing is BGR-in/BGR-out so OpenCV ownership stays explicit."""

    def process(self, bgr: UInt8Frame, timestamp: float) -> ProcessedFrame: ...


class OutputSink(Protocol):
    """Outputs receive packed RGB frames."""

    def send(self, rgb: UInt8Frame) -> None: ...

    def close(self) -> None: ...
