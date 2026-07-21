"""Apple Vision landmark backend used when MediaPipe cannot run."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np

from eyeline.contracts import FaceGeometry, UInt8Frame
from eyeline.landmarks.base import LandmarkBackendUnavailable
from eyeline.landmarks.geometry import eye_aspect_ratio, six_canonical_points


class AppleVisionLandmarkBackend:
    """Run ``VNDetectFaceLandmarksRequest`` directly through PyObjC."""

    def __init__(
        self,
        *,
        vision_module: Any | None = None,
        quartz_module: Any | None = None,
        objc_module: Any | None = None,
    ) -> None:
        try:
            if vision_module is None:
                import Vision as vision_module  # type: ignore[no-redef]
            if quartz_module is None:
                import Quartz as quartz_module  # type: ignore[no-redef]
            if objc_module is None:
                import objc as objc_module  # type: ignore[no-redef]
        except Exception as exc:
            raise LandmarkBackendUnavailable(f"Apple Vision is unavailable: {exc}") from exc
        self._vision = vision_module
        self._quartz = quartz_module
        self._objc = objc_module

    def detect(self, rgb: UInt8Frame) -> FaceGeometry | None:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Apple Vision expects an HxWx3 uint8 RGB frame")

        pool_factory = getattr(self._objc, "autorelease_pool", None)
        pool = pool_factory() if pool_factory is not None else nullcontext()
        with pool:
            cg_image = self._make_cg_image(np.ascontiguousarray(rgb))
            request = self._vision.VNDetectFaceLandmarksRequest.alloc().init()
            handler = self._vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
                cg_image, {}
            )
            performed = handler.performRequests_error_([request], None)
            if isinstance(performed, tuple) and not performed[0]:
                raise RuntimeError(f"Vision request failed: {performed[1]}")
            observations = list(request.results() or [])
            if not observations:
                return None
            observation = max(observations, key=lambda item: float(item.confidence()))
            return self._to_geometry(observation, rgb.shape[1], rgb.shape[0])

    def _make_cg_image(self, rgb: UInt8Frame) -> Any:
        quartz = self._quartz
        height, width = rgb.shape[:2]
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
        payload = rgba.tobytes()
        provider = quartz.CGDataProviderCreateWithData(None, payload, len(payload), None)
        color_space = quartz.CGColorSpaceCreateDeviceRGB()
        bitmap_info = quartz.kCGBitmapByteOrderDefault | quartz.kCGImageAlphaLast
        return quartz.CGImageCreate(
            width,
            height,
            8,
            32,
            width * 4,
            color_space,
            bitmap_info,
            provider,
            None,
            False,
            quartz.kCGRenderingIntentDefault,
        )

    def _to_geometry(self, observation: Any, width: int, height: int) -> FaceGeometry | None:
        face_landmarks = observation.landmarks()
        if face_landmarks is None:
            return None
        all_points = self._region_points(face_landmarks.allPoints(), observation)
        left_eye = six_canonical_points(self._region_points(face_landmarks.leftEye(), observation))
        right_eye = six_canonical_points(
            self._region_points(face_landmarks.rightEye(), observation)
        )
        if not len(left_eye) or not len(right_eye):
            return None

        yaw = math.degrees(self._number(observation, "yaw"))
        pitch = math.degrees(self._number(observation, "pitch"))
        roll = math.degrees(self._number(observation, "roll"))
        return FaceGeometry(
            landmarks=all_points,
            left_eye=left_eye,
            right_eye=right_eye,
            confidence=float(np.clip(observation.confidence(), 0.0, 1.0)),
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            roll_degrees=roll,
            left_eye_openness=float(np.clip(eye_aspect_ratio(left_eye, width, height), 0, 1)),
            right_eye_openness=float(np.clip(eye_aspect_ratio(right_eye, width, height), 0, 1)),
            metadata={"backend": "apple_vision"},
        )

    @staticmethod
    def _number(observation: Any, name: str) -> float:
        accessor = getattr(observation, name, None)
        if accessor is None:
            return 0.0
        value = accessor() if callable(accessor) else accessor
        if value is None:
            return 0.0
        if hasattr(value, "doubleValue"):
            return float(value.doubleValue())
        return float(value)

    @staticmethod
    def _region_points(region: Any, observation: Any) -> np.ndarray:
        if region is None:
            return np.empty((0, 3), dtype=np.float32)
        bbox = observation.boundingBox()
        converted = []
        for point in region.normalizedPoints():
            x = float(bbox.origin.x + point.x * bbox.size.width)
            # Vision uses a bottom-left origin; EyeLine/OpenCV use top-left.
            y = 1.0 - float(bbox.origin.y + point.y * bbox.size.height)
            converted.append((x, y, 0.0))
        return np.asarray(converted, dtype=np.float32)
