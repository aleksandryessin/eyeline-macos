from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eyeline.obs.capture import CameraOpenError, OpenCVCapture


class FakeDevice:
    def __init__(self, opened: bool = True, frame: np.ndarray | None = None) -> None:
        self.opened = opened
        self.frame = frame if frame is not None else np.zeros((2, 3, 3), dtype=np.uint8)
        self.released = False
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def set(self, key: int, value: float) -> bool:
        self.settings.append((key, value))
        return True

    def read(self):
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
