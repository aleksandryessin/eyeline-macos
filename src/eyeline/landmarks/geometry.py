"""Geometry helpers shared by MediaPipe and Apple Vision detectors."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def eye_aspect_ratio(points: NDArray[np.float32], width: int, height: int) -> float:
    """Return a scale-independent eye aspect ratio for six canonical points."""

    if len(points) < 6:
        return 1.0
    xy = np.asarray(points[:6, :2], dtype=np.float32) * np.array([width, height])
    horizontal = float(np.linalg.norm(xy[0] - xy[3]))
    if horizontal < 1e-6:
        return 0.0
    vertical = float(np.linalg.norm(xy[1] - xy[5]) + np.linalg.norm(xy[2] - xy[4]))
    # Typical open eyes land near 0.25-0.35. Scale to the FaceGeometry 0..1 convention.
    return float(np.clip(vertical / (2.0 * horizontal) / 0.30, 0.0, 1.5))


def approximate_head_pose(
    landmarks: NDArray[np.float32], left_eye: NDArray[np.float32], right_eye: NDArray[np.float32]
) -> tuple[float, float, float]:
    """Estimate yaw/pitch/roll from Face Landmarker normalized coordinates."""

    if len(landmarks) <= 454:
        return 0.0, 0.0, _roll(left_eye, right_eye)

    face_left = landmarks[234, :2]
    face_right = landmarks[454, :2]
    forehead = landmarks[10, :2]
    chin = landmarks[152, :2]
    nose = landmarks[1, :2]
    face_mid = (face_left + face_right) * 0.5
    half_width = max(float(np.linalg.norm(face_right - face_left)) * 0.5, 1e-6)
    half_height = max(float(np.linalg.norm(chin - forehead)) * 0.5, 1e-6)

    yaw = float(np.clip((nose[0] - face_mid[0]) / half_width * 42.0, -60.0, 60.0))
    pitch = float(np.clip((nose[1] - face_mid[1]) / half_height * 35.0, -50.0, 50.0))
    return yaw, pitch, _roll(left_eye, right_eye)


def _roll(left_eye: NDArray[np.float32], right_eye: NDArray[np.float32]) -> float:
    if not len(left_eye) or not len(right_eye):
        return 0.0
    delta = np.mean(left_eye[:, :2], axis=0) - np.mean(right_eye[:, :2], axis=0)
    return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))


def six_canonical_points(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Reduce an arbitrary eye contour to a stable six-point representation."""

    array = np.asarray(points, dtype=np.float32)
    if len(array) == 6:
        return array.copy()
    if len(array) < 2:
        return np.empty((0, 3), dtype=np.float32)
    indices = np.linspace(0, len(array) - 1, 6).round().astype(int)
    return array[indices].copy()
