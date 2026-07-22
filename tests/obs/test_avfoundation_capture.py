from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from eyeline.obs.avfoundation_capture import (
    AVFoundationCapture,
    _make_frame_delegate,
    _sample_buffer_to_bgr,
    _select_physical_builtin_camera,
)
from eyeline.obs.capture import CameraOpenError, CameraReadError


class FakeDevice:
    def __init__(self, *, continuity: bool = False, device_type: str = "builtin-wide") -> None:
        self.continuity = continuity
        self.device_type = device_type

    def isContinuityCamera(self) -> bool:
        return self.continuity

    def deviceType(self) -> str:
        return self.device_type


def fake_avfoundation(device: FakeDevice | None):
    class DeviceClass:
        @staticmethod
        def defaultDeviceWithDeviceType_mediaType_position_(*args):
            return device

        @staticmethod
        def devicesWithMediaType_(*args):
            raise AssertionError("direct capture must never enumerate dynamic camera indexes")

    return SimpleNamespace(
        AVCaptureDevice=DeviceClass,
        AVCaptureDeviceTypeBuiltInWideAngleCamera="builtin-wide",
        AVMediaTypeVideo="video",
        AVCaptureDevicePositionUnspecified=0,
    )


def test_selector_returns_exact_builtin_device_without_enumeration() -> None:
    expected = FakeDevice()

    assert _select_physical_builtin_camera(fake_avfoundation(expected)) is expected


def test_selector_refuses_continuity_camera_even_if_framework_returns_it() -> None:
    with pytest.raises(CameraOpenError, match="Refusing"):
        _select_physical_builtin_camera(fake_avfoundation(FakeDevice(continuity=True)))


def test_selector_refuses_wrong_device_type() -> None:
    with pytest.raises(CameraOpenError, match="did not return"):
        _select_physical_builtin_camera(fake_avfoundation(FakeDevice(device_type="external")))


def test_read_has_bounded_startup_timeout() -> None:
    capture = AVFoundationCapture(16, 8, 30, startup_timeout_seconds=0.001)
    capture._session = SimpleNamespace(isRunning=lambda: True)

    with pytest.raises(CameraReadError, match="returned no frame"):
        capture.read()


def test_delegate_class_can_be_reused_across_capture_instances() -> None:
    first_owner = AVFoundationCapture(16, 8, 30)
    second_owner = AVFoundationCapture(16, 8, 30)

    first = _make_frame_delegate(first_owner)
    second = _make_frame_delegate(second_owner)

    assert type(first) is type(second)
    assert first._eyeline_owner is first_owner
    assert second._eyeline_owner is second_owner


def test_sample_buffer_copy_preserves_bgr_and_ignores_row_padding(monkeypatch) -> None:
    width, height, bytes_per_row = 2, 1, 12
    raw = bytearray([10, 20, 30, 255, 40, 50, 60, 255, 99, 99, 99, 99])

    class BaseAddress:
        def as_buffer(self, length):
            assert length == len(raw)
            return memoryview(raw)

    fake_core_media = SimpleNamespace(CMSampleBufferGetImageBuffer=lambda sample: "pixel")
    fake_quartz = SimpleNamespace(
        kCVPixelBufferLock_ReadOnly=1,
        CVPixelBufferLockBaseAddress=lambda pixel, flags: 0,
        CVPixelBufferGetWidth=lambda pixel: width,
        CVPixelBufferGetHeight=lambda pixel: height,
        CVPixelBufferGetBytesPerRow=lambda pixel: bytes_per_row,
        CVPixelBufferGetBaseAddress=lambda pixel: BaseAddress(),
        CVPixelBufferUnlockBaseAddress=lambda pixel, flags: 0,
    )
    monkeypatch.setitem(sys.modules, "CoreMedia", fake_core_media)
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    frame = _sample_buffer_to_bgr("sample")

    np.testing.assert_array_equal(frame, np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8))
