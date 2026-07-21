"""Temporal correction gating that keeps gaze correction natural and stable."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from eyeline.config import NaturalizerConfig
from eyeline.contracts import FaceGeometry, FloatPoints


@dataclass(frozen=True, slots=True)
class NaturalizerResult:
    geometry: FaceGeometry | None
    effective_strength: float
    reason: str | None = None


class GazeNaturalizer:
    """Smooth landmark motion and fade correction around unsafe facial states."""

    def __init__(
        self,
        config: NaturalizerConfig,
        *,
        correction_strength: float,
        min_confidence: float,
        max_point_step: float = 0.035,
    ) -> None:
        self.config = config
        self.correction_strength = float(np.clip(correction_strength, 0.0, 1.0))
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.max_point_step = max_point_step
        self._level = 0.0
        self._last_timestamp: float | None = None
        self._smoothed: FaceGeometry | None = None

    def update(self, geometry: FaceGeometry | None, timestamp: float) -> NaturalizerResult:
        dt = self._delta_time(timestamp)
        if geometry is None:
            self._fade_toward(0.0, dt)
            self._smoothed = None
            return NaturalizerResult(None, 0.0, "no_face")

        smoothed = self._smooth_geometry(geometry)
        target, reason = self._gate(geometry)
        self._fade_toward(target, dt)
        effective = self.correction_strength * self._level
        if effective <= 1e-4:
            effective = 0.0
        return NaturalizerResult(smoothed, effective, reason if effective == 0.0 else None)

    def reset(self) -> None:
        self._level = 0.0
        self._last_timestamp = None
        self._smoothed = None

    def _delta_time(self, timestamp: float) -> float:
        if self._last_timestamp is None or timestamp <= self._last_timestamp:
            dt = 0.0
        else:
            # A suspended process should not jump instantly to a new correction level.
            dt = min(timestamp - self._last_timestamp, 0.25)
        self._last_timestamp = timestamp
        return dt

    def _gate(self, geometry: FaceGeometry) -> tuple[float, str | None]:
        if (
            geometry.left_eye_openness < self.config.blink_threshold
            or geometry.right_eye_openness < self.config.blink_threshold
        ):
            return 0.0, "blink"

        confidence_span = max(1.0 - self.min_confidence, 1e-6)
        confidence_factor = float(
            np.clip((geometry.confidence - self.min_confidence) / confidence_span, 0.0, 1.0)
        )
        if confidence_factor <= 0.0:
            return 0.0, "low_confidence"

        head_angle = max(abs(geometry.yaw_degrees), abs(geometry.pitch_degrees))
        soft = self.config.head_turn_soft_limit_degrees
        hard = max(self.config.head_turn_hard_limit_degrees, soft + 1e-6)
        head_factor = float(np.clip((hard - head_angle) / (hard - soft), 0.0, 1.0))
        if head_angle <= soft:
            head_factor = 1.0
        if head_factor <= 0.0:
            return 0.0, "head_turn"

        gaze_threshold = self.config.extreme_gaze_threshold
        gaze_factor = float(
            np.clip((1.0 - geometry.gaze_extremity) / max(1.0 - gaze_threshold, 1e-6), 0, 1)
        )
        if geometry.gaze_extremity <= gaze_threshold:
            gaze_factor = 1.0
        if gaze_factor <= 0.0:
            return 0.0, "extreme_gaze"

        return confidence_factor * head_factor * gaze_factor, None

    def _fade_toward(self, target: float, dt: float) -> None:
        target = float(np.clip(target, 0.0, 1.0))
        seconds = (
            self.config.fade_in_seconds if target > self._level else self.config.fade_out_seconds
        )
        if seconds <= 0.0:
            self._level = target
            return
        maximum_change = dt / seconds
        self._level += float(np.clip(target - self._level, -maximum_change, maximum_change))
        self._level = float(np.clip(self._level, 0.0, 1.0))

    def _smooth_geometry(self, current: FaceGeometry) -> FaceGeometry:
        previous = self._smoothed
        if previous is None or previous.landmarks.shape != current.landmarks.shape:
            self._smoothed = current
            return current
        alpha = self.config.smoothing_alpha
        smoothed = replace(
            current,
            landmarks=self._smooth_points(previous.landmarks, current.landmarks, alpha),
            left_eye=self._smooth_points(previous.left_eye, current.left_eye, alpha),
            right_eye=self._smooth_points(previous.right_eye, current.right_eye, alpha),
            yaw_degrees=self._ema(previous.yaw_degrees, current.yaw_degrees, alpha),
            pitch_degrees=self._ema(previous.pitch_degrees, current.pitch_degrees, alpha),
            roll_degrees=self._ema(previous.roll_degrees, current.roll_degrees, alpha),
            # Keep current blink/gaze signals unsmoothed so eyelid motion is never erased.
            left_eye_openness=current.left_eye_openness,
            right_eye_openness=current.right_eye_openness,
            gaze_extremity=current.gaze_extremity,
        )
        self._smoothed = smoothed
        return smoothed

    def _smooth_points(
        self, previous: FloatPoints, current: FloatPoints, alpha: float
    ) -> FloatPoints:
        if previous.shape != current.shape:
            return current
        delta = np.asarray(current - previous, dtype=np.float32)
        if delta.shape[-1] >= 2:
            distance = np.linalg.norm(delta[..., :2], axis=-1, keepdims=True)
            scale = np.minimum(1.0, self.max_point_step / np.maximum(distance, 1e-8))
            delta[..., :2] *= scale
        return np.asarray(previous + alpha * delta, dtype=np.float32)

    @staticmethod
    def _ema(previous: float, current: float, alpha: float) -> float:
        return float(previous + alpha * (current - previous))
