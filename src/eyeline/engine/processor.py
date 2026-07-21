"""Fail-open BGR frame processor."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from eyeline.contracts import FaceGeometry, LandmarkBackend, ProcessedFrame, UInt8Frame
from eyeline.naturalizer import GazeNaturalizer


class GazeCorrector(Protocol):
    def correct(
        self, frame_bgr: UInt8Frame, geometry: FaceGeometry, strength: float
    ) -> UInt8Frame: ...

    def close(self) -> None: ...


def bgr_to_rgb(frame_bgr: UInt8Frame) -> UInt8Frame:
    """Return packed RGB data; never relabel BGR memory as SRGB."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("frame must be HxWx3 uint8 BGR")
    return np.ascontiguousarray(frame_bgr[..., ::-1])


class EyeLineFrameProcessor:
    """Detect, naturalize, and correct a frame while guaranteeing original fallback."""

    def __init__(
        self,
        landmarks: LandmarkBackend,
        corrector: GazeCorrector,
        naturalizer: GazeNaturalizer,
    ) -> None:
        self.landmarks = landmarks
        self.corrector = corrector
        self.naturalizer = naturalizer

    def process(self, bgr: UInt8Frame, timestamp: float) -> ProcessedFrame:
        started = time.perf_counter()
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("frame must be HxWx3 uint8 BGR")
        original = np.ascontiguousarray(bgr).copy()
        try:
            geometry = self.landmarks.detect(bgr_to_rgb(original))
        except Exception as exc:
            return self._fallback(
                original,
                timestamp,
                started,
                False,
                f"landmark_error:{type(exc).__name__}",
            )

        if geometry is None:
            self.naturalizer.update(None, timestamp)
            return self._fallback(original, timestamp, started, False, "no_face")

        decision = self.naturalizer.update(geometry, timestamp)
        if decision.geometry is None or decision.effective_strength <= 0.0:
            return self._fallback(
                original,
                timestamp,
                started,
                True,
                decision.reason or "correction_gated",
                decision.effective_strength,
            )
        try:
            corrected = self.corrector.correct(
                original, decision.geometry, decision.effective_strength
            )
            if (
                corrected.shape != original.shape
                or corrected.dtype != np.uint8
                or corrected.ndim != 3
            ):
                raise ValueError("corrector returned an invalid frame")
            return ProcessedFrame(
                frame_bgr=np.ascontiguousarray(corrected),
                timestamp=timestamp,
                face_detected=True,
                correction_applied=True,
                effective_strength=decision.effective_strength,
                processing_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            return self._fallback(
                original,
                timestamp,
                started,
                True,
                f"correction_error:{type(exc).__name__}",
                0.0,
            )

    @staticmethod
    def _fallback(
        frame: UInt8Frame,
        timestamp: float,
        started: float,
        face_detected: bool,
        reason: str,
        effective_strength: float = 0.0,
    ) -> ProcessedFrame:
        return ProcessedFrame(
            frame_bgr=frame,
            timestamp=timestamp,
            face_detected=face_detected,
            correction_applied=False,
            effective_strength=effective_strength,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            reason=reason,
        )

    def close(self) -> None:
        for resource in (self.landmarks, self.corrector):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
