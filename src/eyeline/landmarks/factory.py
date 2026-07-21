"""Landmark backend selection."""

from __future__ import annotations

import logging
from pathlib import Path

from eyeline.contracts import LandmarkBackend
from eyeline.landmarks.apple_vision import AppleVisionLandmarkBackend
from eyeline.landmarks.base import AutoLandmarkBackend, NullLandmarkBackend
from eyeline.landmarks.mediapipe_tasks import MediaPipeTasksLandmarkBackend

LOG = logging.getLogger(__name__)


def create_landmark_backend(
    backend: str,
    *,
    model_path: str | Path,
    min_confidence: float = 0.55,
) -> LandmarkBackend:
    """Create ``mediapipe``, ``vision``, or resilient ``auto`` detection."""

    normalized = backend.strip().lower()
    if normalized in {"mediapipe", "mediapipe_tasks"}:
        return MediaPipeTasksLandmarkBackend(model_path, min_confidence=min_confidence)
    if normalized in {"vision", "apple_vision"}:
        return AppleVisionLandmarkBackend()
    if normalized != "auto":
        raise ValueError(f"unsupported landmark backend: {backend}")

    primary = None
    fallback = None
    try:
        primary = MediaPipeTasksLandmarkBackend(model_path, min_confidence=min_confidence)
    except Exception:
        LOG.exception("MediaPipe could not initialize; Apple Vision will be primary at runtime")
    try:
        fallback = AppleVisionLandmarkBackend()
    except Exception:
        LOG.exception("Apple Vision could not initialize")
    if primary is None and fallback is None:
        return NullLandmarkBackend()
    return AutoLandmarkBackend(primary, fallback)
