"""High-level eye extraction and TensorFlow gaze correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from eyeline.contracts import FaceGeometry, UInt8Frame
from eyeline.models.tensorflow_checkpoint import TensorFlowEyeModel


class EyeInferenceModel(Protocol):
    def infer_eye(
        self,
        side: str,
        image_bgr: NDArray[np.float32],
        anchor_map: NDArray[np.float32],
        angles: tuple[float, float],
    ) -> NDArray[np.float32]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EyeCrop:
    image_bgr: NDArray[np.float32]
    anchor_map: NDArray[np.float32]
    blend_mask: NDArray[np.float32]
    bounds: tuple[int, int, int, int]  # top, bottom, left, right
    center: tuple[float, float]


class TensorFlowGazeCorrector:
    """Apply the upstream DeepWarp checkpoint to both eye regions."""

    def __init__(
        self,
        checkpoint_root: str | Path | None = None,
        *,
        model: EyeInferenceModel | None = None,
        focal_length: float = 650.0,
        ipd_cm: float = 6.3,
        camera_offset: tuple[float, float, float] = (0.0, -21.0, -1.0),
    ) -> None:
        if model is None:
            if checkpoint_root is None:
                raise ValueError("checkpoint_root is required when model is not injected")
            model = TensorFlowEyeModel(checkpoint_root)
        self.model = model
        self.focal_length = focal_length
        self.ipd_cm = ipd_cm
        self.camera_offset = camera_offset

    def correct(self, frame_bgr: UInt8Frame, geometry: FaceGeometry, strength: float) -> UInt8Frame:
        """Return a corrected BGR frame without mutating the input."""

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return frame_bgr.copy()
        left = extract_eye_crop(frame_bgr, geometry.left_eye, "L")
        right = extract_eye_crop(frame_bgr, geometry.right_eye, "R")
        if left is None or right is None:
            raise ValueError("eye crop lies outside the frame")
        height, width = frame_bgr.shape[:2]
        # Infer the complete camera-target redirection. ``strength`` is applied once,
        # while compositing the learned delta below. Scaling both the target angles and
        # the delta made a configured 0.55 correction behave roughly like 0.55².
        angles = self._estimate_angles(left.center, right.center, (width, height))
        output = frame_bgr.copy()
        for side, crop in (("L", left), ("R", right)):
            predicted = self.model.infer_eye(side, crop.image_bgr, crop.anchor_map, angles)
            self._composite(output, crop, predicted, strength)
        return output

    def _estimate_angles(
        self,
        left_center: tuple[float, float],
        right_center: tuple[float, float],
        frame_size: tuple[int, int],
    ) -> tuple[float, float]:
        ipd_pixels = max(math.dist(left_center, right_center), 1.0)
        eye_z = -(self.focal_length * self.ipd_cm) / ipd_pixels
        eye_x = (
            -abs(eye_z)
            * (left_center[0] + right_center[0] - frame_size[0])
            / (2.0 * self.focal_length)
            + self.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (left_center[1] + right_center[1] - frame_size[1])
            / (2.0 * self.focal_length)
            + self.camera_offset[1]
        )
        vertical = math.degrees(math.atan2(-eye_y, -eye_z))
        horizontal = math.degrees(math.atan2(-eye_x, -eye_z))
        vertical += math.degrees(
            math.atan2(eye_y - self.camera_offset[1], self.camera_offset[2] - eye_z)
        )
        horizontal += math.degrees(
            math.atan2(eye_x - self.camera_offset[0], self.camera_offset[2] - eye_z)
        )
        # The checkpoint is trained for modest redirections; bounding avoids torn crops.
        return (
            float(np.clip(vertical, -15.0, 15.0)),
            float(np.clip(horizontal, -15.0, 15.0)),
        )

    @staticmethod
    def _composite(
        output: UInt8Frame,
        crop: EyeCrop,
        predicted_bgr: NDArray[np.float32],
        strength: float,
    ) -> None:
        top, bottom, left, right = crop.bounds
        target_height, target_width = bottom - top, right - left
        predicted = np.asarray(predicted_bgr, dtype=np.float32)
        if predicted.shape != (48, 64, 3) or not np.isfinite(predicted).all():
            raise ValueError("gaze model returned an invalid eye image")
        # Upscaling the model's complete 64x48 result would blur the original HD eye
        # crop. Apply only the learned correction delta so native camera detail survives.
        delta = predicted - crop.image_bgr
        delta = cv2.resize(
            delta, (target_width, target_height), interpolation=cv2.INTER_CUBIC
        )
        original = output[top:bottom, left:right].astype(np.float32)
        alpha = np.asarray(crop.blend_mask, dtype=np.float32)[..., None] * strength
        blended = original + delta * 255.0 * alpha
        output[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)

    def close(self) -> None:
        self.model.close()


def extract_eye_crop(
    frame_bgr: UInt8Frame,
    normalized_points: NDArray[np.float32],
    side: str,
    *,
    input_size: tuple[int, int] = (48, 64),
) -> EyeCrop | None:
    """Extract a BGR eye crop and the checkpoint's 12-channel anchor map."""

    points = np.asarray(normalized_points, dtype=np.float32)
    if points.ndim != 2 or len(points) < 6 or points.shape[1] < 2:
        return None
    height, width = frame_bgr.shape[:2]
    pixels = points[:6, :2] * np.array([width, height], dtype=np.float32)
    center = np.mean(pixels, axis=0)
    eye_width = float(np.max(pixels[:, 0]) - np.min(pixels[:, 0]))
    if eye_width < 4.0:
        return None
    half_width = eye_width * 0.75
    box_height = 1.5 * half_width
    top = max(0, int(round(center[1] - box_height * 7.0 / 12.0)))
    bottom = min(height, int(round(center[1] + box_height * 5.0 / 12.0)))
    left = max(0, int(round(center[0] - half_width)))
    right = min(width, int(round(center[0] + half_width)))
    if bottom - top < 4 or right - left < 4:
        return None

    source = frame_bgr[top:bottom, left:right]
    resized = cv2.resize(source, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    image_bgr = np.asarray(resized, dtype=np.float32) / 255.0
    blend_mask = _eye_blend_mask(pixels, (top, bottom, left, right), eye_width)
    sequence = (3, 2, 1, 0, 5, 4) if side.upper() == "L" else (0, 1, 2, 3, 4, 5)
    grid_y, grid_x = np.mgrid[0 : input_size[0], 0 : input_size[1]]
    anchors = []
    for index in sequence:
        anchor_x = (pixels[index, 0] - left) * input_size[1] / (right - left)
        anchor_y = (pixels[index, 1] - top) * input_size[0] / (bottom - top)
        anchors.extend((grid_x - anchor_x, grid_y - anchor_y))
    anchor_map = np.stack(anchors, axis=2).astype(np.float32)
    return EyeCrop(
        image_bgr,
        anchor_map,
        blend_mask,
        (top, bottom, left, right),
        tuple(center),
    )


def _eye_blend_mask(
    eye_pixels: NDArray[np.float32],
    bounds: tuple[int, int, int, int],
    eye_width: float,
) -> NDArray[np.float32]:
    """Build a soft contour mask that never exposes the rectangular crop boundary."""

    top, bottom, left, right = bounds
    height, width = bottom - top, right - left
    contour = np.rint(
        eye_pixels[:6, :2] - np.array([left, top], dtype=np.float32)
    ).astype(np.int32)
    contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(contour), 255)
    radius = max(2, int(round(eye_width * 0.16)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    mask = cv2.dilate(mask, kernel)
    sigma = max(1.0, eye_width * 0.08)
    softened = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = softened.astype(np.float32) / 255.0

    # Gaussian tails can still touch the crop edge. Force a zero-valued border and
    # smoothly reach full opacity inside it, eliminating rectangular seams.
    y_distance = np.minimum(np.arange(height), np.arange(height)[::-1])
    x_distance = np.minimum(np.arange(width), np.arange(width)[::-1])
    edge_distance = np.minimum(y_distance[:, None], x_distance[None, :]).astype(np.float32)
    feather = max(2.0, min(height, width) * 0.10)
    edge_taper = np.clip(edge_distance / feather, 0.0, 1.0)
    return np.ascontiguousarray(np.clip(normalized * edge_taper, 0.0, 1.0))
