"""OpenCV camera input with deterministic cleanup."""

from __future__ import annotations

import platform
import time
from types import TracebackType
from typing import Any

import numpy as np


class CameraOpenError(RuntimeError):
    """Raised when the configured physical camera cannot be opened."""


class CameraReadError(RuntimeError):
    """Raised after a camera backend fails to return a usable frame."""


class OpenCVCapture:
    """Capture BGR frames at the requested size, releasing the device reliably."""

    def __init__(
        self,
        index: int | None,
        width: int,
        height: int,
        fps: int,
        *,
        cv2_module: Any | None = None,
        read_attempts: int = 30,
        read_retry_seconds: float = 0.05,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if read_attempts <= 0 or read_retry_seconds < 0:
            raise ValueError("read attempts must be positive and retry delay non-negative")
        if cv2_module is None:
            import cv2 as cv2_module

        self._cv2 = cv2_module
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.read_attempts = read_attempts
        self.read_retry_seconds = read_retry_seconds
        self._capture: Any | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    def open(self) -> OpenCVCapture:
        if self.is_open:
            return self

        camera_index = 0 if self.index is None else self.index
        api = getattr(self._cv2, "CAP_AVFOUNDATION", None)
        if platform.system() == "Darwin" and api is not None:
            capture = self._cv2.VideoCapture(camera_index, api)
        else:
            capture = self._cv2.VideoCapture(camera_index)
        self._capture = capture

        if not capture.isOpened():
            self.close()
            camera_name = (
                "the built-in Mac camera"
                if self.index is None
                else f"camera index {self.index}"
            )
            raise CameraOpenError(
                f"Cannot open {camera_name}. Allow Camera access in "
                "System Settings > Privacy & Security > Camera, then retry."
            )

        capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        capture.set(self._cv2.CAP_PROP_FPS, float(self.fps))
        buffersize = getattr(self._cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffersize is not None:
            capture.set(buffersize, 1.0)
        return self

    def read(self) -> np.ndarray:
        if not self.is_open:
            raise CameraReadError("camera is not open")
        frame = None
        for attempt in range(self.read_attempts):
            ok, candidate = self._capture.read()
            if (
                ok
                and candidate is not None
                and candidate.ndim == 3
                and candidate.shape[2] == 3
            ):
                frame = candidate
                break
            if attempt + 1 < self.read_attempts and self.read_retry_seconds:
                time.sleep(self.read_retry_seconds)
        if frame is None:
            raise CameraReadError(f"Camera index {self.index} did not return a BGR frame")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[:2] != (self.height, self.width):
            frame = self._cv2.resize(frame, (self.width, self.height))
        return np.ascontiguousarray(frame)

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> OpenCVCapture:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def create_camera_capture(index: int | None, width: int, height: int, fps: int):
    """Create a stable direct macOS capture, or indexed OpenCV only when explicitly requested."""

    if platform.system() == "Darwin" and index is None:
        from eyeline.obs.avfoundation_capture import AVFoundationCapture

        return AVFoundationCapture(width, height, fps)
    return OpenCVCapture(index, width, height, fps)
