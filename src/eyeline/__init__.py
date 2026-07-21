"""EyeLine local gaze-correction pipeline."""

from .config import EyeLineConfig, load_config
from .contracts import FaceGeometry, FrameProcessor, LandmarkBackend, OutputSink, ProcessedFrame

__all__ = [
    "EyeLineConfig",
    "FaceGeometry",
    "FrameProcessor",
    "LandmarkBackend",
    "OutputSink",
    "ProcessedFrame",
    "load_config",
]
