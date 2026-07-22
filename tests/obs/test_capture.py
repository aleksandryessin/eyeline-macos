from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eyeline.obs.avfoundation_capture import AVFoundationCapture
from eyeline.obs.capture import (
    CameraOpenError,
    CameraReadError,
    OpenCVCapture,
    create_camera_capture,
)


class FakeDevice:
    def __init__(
        self,
        opened: bool = True,
        frame: np.ndarray | None = None,
        fail_reads: int = 0,
    ) -> None:
        self.opened = opened
        self.frame = frame if frame is not None else np.zeros((2, 3, 3), dtype=np.uint8)
        self.released = False
        self.fail_reads = fail_reads
        self.read_calls = 0
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def set(self, key: int, value: float) -> bool:
        self.settings.append((key, value))
        return True

    def read(self):
        self.read_calls += 1
        if self.read_calls <= self.fail_reads:
            return False, None
        return True, self.frame

    def release(self) -> None:
        self.released = True


def fake_cv2(device: FakeDevice):
    return SimpleNamespace(
        CAP_AVFOUNDATION=1200,
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
        CAP_PROP_FPS=3,
        CAP_PROP_BUFFERSIZE=4,
        VideoCapture=lambda *args: device,
        resize=lambda frame, size: np.zeros((size[1], size[0], 3), dtype=np.uint8),
    )


def test_capture_configures_resizes_and_releases() -> None:
    device = FakeDevice()
    capture = OpenCVCapture(0, 8, 6, 30, cv2_module=fake_cv2(device))
    with capture as opened:
        assert opened.read().shape == (6, 8, 3)
        assert (1, 8.0) in device.settings
        assert (2, 6.0) in device.settings
        assert (3, 30.0) in device.settings
    assert device.released


def test_failed_open_still_releases_device() -> None:
    device = FakeDevice(opened=False)
    capture = OpenCVCapture(4, 8, 6, 30, cv2_module=fake_cv2(device))
    with pytest.raises(CameraOpenError, match="Camera access"):
        capture.open()
    assert device.released


def test_capture_retries_transient_startup_reads() -> None:
    device = FakeDevice(fail_reads=2)
    capture = OpenCVCapture(
        0,
        8,
        6,
        30,
        cv2_module=fake_cv2(device),
        read_attempts=3,
        read_retry_seconds=0,
    )
    with capture as opened:
        assert opened.read().shape == (6, 8, 3)
    assert device.read_calls == 3


def test_capture_stops_after_bounded_read_retries() -> None:
    device = FakeDevice(fail_reads=3)
    capture = OpenCVCapture(
        0,
        8,
        6,
        30,
        cv2_module=fake_cv2(device),
        read_attempts=3,
        read_retry_seconds=0,
    )
    with capture as opened, pytest.raises(CameraReadError, match="did not return"):
        opened.read()
    assert device.read_calls == 3


def test_default_macos_capture_bypasses_opencv_indexes(monkeypatch) -> None:
    monkeypatch.setattr("eyeline.obs.capture.platform.system", lambda: "Darwin")

    capture = create_camera_capture(None, 1920, 1080, 30)

    assert isinstance(capture, AVFoundationCapture)


def test_explicit_index_is_the_only_macos_opencv_escape_hatch(monkeypatch) -> None:
    monkeypatch.setattr("eyeline.obs.capture.platform.system", lambda: "Darwin")

    capture = create_camera_capture(3, 1920, 1080, 30)

    assert isinstance(capture, OpenCVCapture)
    assert capture.index == 3
