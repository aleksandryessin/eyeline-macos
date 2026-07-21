"""Face landmark implementations."""

from eyeline.landmarks.apple_vision import AppleVisionLandmarkBackend
from eyeline.landmarks.base import (
    AutoLandmarkBackend,
    LandmarkBackendUnavailable,
    NullLandmarkBackend,
)
from eyeline.landmarks.factory import create_landmark_backend
from eyeline.landmarks.mediapipe_tasks import MediaPipeTasksLandmarkBackend

__all__ = [
    "AppleVisionLandmarkBackend",
    "AutoLandmarkBackend",
    "LandmarkBackendUnavailable",
    "MediaPipeTasksLandmarkBackend",
    "NullLandmarkBackend",
    "create_landmark_backend",
]
