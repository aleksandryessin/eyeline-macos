"""MediaPipe Tasks Face Landmarker backend (the primary EyeLine detector)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eyeline.contracts import FaceGeometry, UInt8Frame
from eyeline.landmarks.base import LandmarkBackendUnavailable
from eyeline.landmarks.geometry import (
    approximate_head_pose,
    eye_aspect_ratio,
    head_pose_from_transform,
)

# Six-point ordering matches the DeepWarp anchor-map convention.
LEFT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_INDICES = (33, 160, 158, 133, 153, 144)


class MediaPipeTasksLandmarkBackend:
    """Detect one face using ``mediapipe.tasks.vision.FaceLandmarker``.

    The public contract receives RGB. This deliberately fixes the upstream integration
    bug that labelled OpenCV BGR memory as ``SRGB``.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        min_confidence: float = 0.55,
        mp_module: Any | None = None,
        landmarker: Any | None = None,
    ) -> None:
        path = Path(model_path)
        if landmarker is None and not path.is_file():
            raise LandmarkBackendUnavailable(f"MediaPipe model not found: {path}")
        try:
            if mp_module is None:
                import mediapipe as mp_module  # type: ignore[no-redef]

            self._mp = mp_module
            if landmarker is None:
                options = mp_module.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp_module.tasks.BaseOptions(model_asset_path=str(path)),
                    running_mode=mp_module.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=min_confidence,
                    min_face_presence_confidence=min_confidence,
                    min_tracking_confidence=min_confidence,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
                landmarker = mp_module.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise LandmarkBackendUnavailable(
                f"MediaPipe Tasks initialization failed: {exc}"
            ) from exc
        self._landmarker = landmarker
        self._min_confidence = min_confidence

    def detect(self, rgb: UInt8Frame) -> FaceGeometry | None:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("MediaPipe expects an HxWx3 uint8 RGB frame")
        packed_rgb = np.ascontiguousarray(rgb)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=packed_rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        return self._to_geometry(result, packed_rgb.shape[1], packed_rgb.shape[0])

    def _to_geometry(self, result: Any, width: int, height: int) -> FaceGeometry:
        raw = result.face_landmarks[0]
        landmarks = np.asarray(
            [[float(point.x), float(point.y), float(getattr(point, "z", 0.0))] for point in raw],
            dtype=np.float32,
        )
        left_eye = landmarks[np.asarray(LEFT_EYE_INDICES)]
        right_eye = landmarks[np.asarray(RIGHT_EYE_INDICES)]
        pose_source = "landmarks"
        pose = None
        transforms = getattr(result, "facial_transformation_matrixes", None)
        if transforms:
            pose = head_pose_from_transform(transforms[0])
        if pose is None:
            pose = approximate_head_pose(landmarks, left_eye, right_eye)
        else:
            pose_source = "facial_transform"
        yaw, pitch, roll = pose

        confidence_values = []
        for point in raw:
            for attribute in ("presence", "visibility"):
                value = float(getattr(point, attribute, 0.0) or 0.0)
                if value > 0.0:
                    confidence_values.append(value)
        # FaceLandmarker already applies the configured presence/detection thresholds. Some
        # task model versions omit per-landmark confidence entirely.
        confidence = (
            float(np.clip(np.median(confidence_values), 0.0, 1.0)) if confidence_values else 1.0
        )

        blendshapes = self._blendshape_map(result)
        left_open = 1.0 - blendshapes.get(
            "eyeBlinkLeft", 1.0 - eye_aspect_ratio(left_eye, width, height)
        )
        right_open = 1.0 - blendshapes.get(
            "eyeBlinkRight", 1.0 - eye_aspect_ratio(right_eye, width, height)
        )
        look_values = [score for name, score in blendshapes.items() if name.startswith("eyeLook")]
        gaze_extremity = float(np.clip(max(look_values, default=0.0), 0.0, 1.0))

        return FaceGeometry(
            landmarks=landmarks,
            left_eye=left_eye,
            right_eye=right_eye,
            confidence=max(self._min_confidence, confidence),
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            roll_degrees=roll,
            left_eye_openness=float(np.clip(left_open, 0.0, 1.0)),
            right_eye_openness=float(np.clip(right_open, 0.0, 1.0)),
            gaze_extremity=gaze_extremity,
            metadata={
                "backend": "mediapipe_tasks",
                "blendshapes": blendshapes,
                "pose_source": pose_source,
            },
        )

    @staticmethod
    def _blendshape_map(result: Any) -> dict[str, float]:
        if not getattr(result, "face_blendshapes", None):
            return {}
        categories = result.face_blendshapes[0]
        values: dict[str, float] = {}
        for category in categories:
            name = getattr(category, "category_name", None) or getattr(
                category, "display_name", None
            )
            if name:
                values[str(name)] = float(category.score)
        return values

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if close is not None:
            close()
