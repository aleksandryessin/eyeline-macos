from __future__ import annotations

import numpy as np

from eyeline.contracts import ProcessedFrame
from eyeline.obs.fixtures import NullSink, PassthroughProcessor, SyntheticCapture
from eyeline.obs.runner import PipelineRunner


class OneFrameSource:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self) -> np.ndarray:
        return self.frame.copy()


class RaisingProcessor:
    def process(self, bgr: np.ndarray, timestamp: float):
        raise RuntimeError("model failed")


class RedProcessor:
    def process(self, bgr: np.ndarray, timestamp: float) -> ProcessedFrame:
        corrected = np.array([[[0, 0, 255]]], dtype=np.uint8)
        return ProcessedFrame(corrected, timestamp, True, True, 1.0, 1.0)


def test_runner_converts_corrected_bgr_to_explicit_rgb() -> None:
    source = OneFrameSource(np.array([[[255, 0, 0]]], dtype=np.uint8))
    sink = NullSink()
    stats = PipelineRunner(source, RedProcessor(), sink).run(
        max_frames=1, install_signal_handlers=False
    )
    assert sink.last_rgb.tolist() == [[[255, 0, 0]]]
    assert stats.frames_sent == 1
    assert source.closed


def test_processor_exception_fails_open_to_original_frame() -> None:
    blue_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
    sink = NullSink()
    stats = PipelineRunner(OneFrameSource(blue_bgr), RaisingProcessor(), sink).run(
        max_frames=1, install_signal_handlers=False
    )
    assert sink.last_rgb.tolist() == [[[0, 0, 255]]]
    assert stats.processor_failures == 1
    assert stats.frames_sent == 1


def test_synthetic_fixture_soak_does_not_touch_camera_or_obs() -> None:
    sink = NullSink()
    stats = PipelineRunner(
        SyntheticCapture(16, 8, 30, realtime=False), PassthroughProcessor(), sink
    ).run(max_frames=120, install_signal_handlers=False)
    assert stats.frames_sent == 120
    assert stats.processor_failures == 0
    assert sink.last_rgb.shape == (8, 16, 3)
