from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eyeline.obs.sink import OBSVirtualCameraSink, VirtualCameraError


class FakeCamera:
    device = "OBS Virtual Camera"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.frames: list[np.ndarray] = []
        self.sleeps = 0
        self.closed = False

    def send(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def sleep_until_next_frame(self) -> None:
        self.sleeps += 1

    def close(self) -> None:
        self.closed = True


def test_sink_declares_rgb_and_closes() -> None:
    cameras: list[FakeCamera] = []

    def camera_factory(**kwargs):
        camera = FakeCamera(**kwargs)
        cameras.append(camera)
        return camera

    module = SimpleNamespace(PixelFormat=SimpleNamespace(RGB="RGB"), Camera=camera_factory)
    sink = OBSVirtualCameraSink(2, 1, 30, pyvirtualcam_module=module)
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    with sink:
        sink.send(rgb)
        assert sink.device == "OBS Virtual Camera"
    assert cameras[0].kwargs["fmt"] == "RGB"
    assert cameras[0].kwargs["backend"] == "obs"
    assert cameras[0].frames[0].tolist() == rgb.tolist()
    assert cameras[0].sleeps == 1
    assert cameras[0].closed


def test_sink_translates_backend_open_error() -> None:
    def fail(**kwargs):
        raise RuntimeError("backend busy")

    module = SimpleNamespace(PixelFormat=SimpleNamespace(RGB="RGB"), Camera=fail)
    with pytest.raises(VirtualCameraError, match="quit OBS"):
        OBSVirtualCameraSink(2, 1, 30, pyvirtualcam_module=module).open()


def test_sink_rejects_bgr_shape_mismatch() -> None:
    module = SimpleNamespace(PixelFormat=SimpleNamespace(RGB="RGB"), Camera=FakeCamera)
    with OBSVirtualCameraSink(2, 1, 30, pyvirtualcam_module=module) as sink:
        with pytest.raises(ValueError, match="shape"):
            sink.send(np.zeros((2, 2, 3), dtype=np.uint8))
