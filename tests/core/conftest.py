from __future__ import annotations

import numpy as np

from eyeline.contracts import FaceGeometry


def make_geometry(
    *,
    confidence: float = 1.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    openness: float = 1.0,
    gaze: float = 0.0,
    offset: float = 0.0,
) -> FaceGeometry:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[:, :2] = (0.5 + offset, 0.5)
    left = np.array(
        [
            [0.58, 0.45, 0],
            [0.60, 0.44, 0],
            [0.63, 0.44, 0],
            [0.66, 0.45, 0],
            [0.63, 0.46, 0],
            [0.60, 0.46, 0],
        ],
        dtype=np.float32,
    )
    right = left.copy()
    right[:, 0] -= 0.24
    left[:, 0] += offset
    right[:, 0] += offset
    return FaceGeometry(
        landmarks=landmarks,
        left_eye=left,
        right_eye=right,
        confidence=confidence,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        left_eye_openness=openness,
        right_eye_openness=openness,
        gaze_extremity=gaze,
    )
