"""Deterministic local sources and sinks for CI and soak testing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import TracebackType

import numpy as np

from eyeline.contracts import ProcessedFrame


class PassthroughProcessor:
    def process(self, bgr: np.ndarray, timestamp: float) -> ProcessedFrame:
        return ProcessedFrame(bgr, timestamp, False, False, 0.0, 0.0, "passthrough")


class SyntheticCapture:
    """Generate moving BGR test frames without touching a physical camera."""

    def __init__(self, width: int, height: int, fps: int, *, realtime: bool = True) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.realtime = realtime
        self._frame_number = 0
        self._deadline = 0.0

    def __enter__(self) -> SyntheticCapture:
        self._deadline = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> np.ndarray:
        if self.realtime:
            delay = self._deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._deadline = max(self._deadline + 1.0 / self.fps, time.monotonic())
        x = np.arange(self.width, dtype=np.uint16)[None, :]
        y = np.arange(self.height, dtype=np.uint16)[:, None]
        phase = self._frame_number % 256
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[..., 0] = (x + phase) % 256  # blue
        frame[..., 1] = (y + phase * 2) % 256  # green
        frame[..., 2] = ((x // 2 + y // 2) + phase * 3) % 256  # red
        self._frame_number += 1
        return frame


@dataclass
class NullSink:
    frames_sent: int = 0
    last_rgb: np.ndarray | None = None

    def __enter__(self) -> NullSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def send(self, rgb: np.ndarray) -> None:
        self.frames_sent += 1
        self.last_rgb = rgb

    def close(self) -> None:
        return None
