from __future__ import annotations

from types import SimpleNamespace

import pytest

from eyeline.obs.macos_camera import BuiltInCameraUnavailable, resolve_builtin_camera_index


class FakeDevice:
    def __init__(self, unique_id: str, *, continuity: bool = False) -> None:
        self._unique_id = unique_id
        self._continuity = continuity

    def uniqueID(self) -> str:
        return self._unique_id

    def isContinuityCamera(self) -> bool:
        return self._continuity

    def localizedName(self) -> str:
        return self._unique_id


def fake_avfoundation(*, target: FakeDevice | None, video_devices: list[FakeDevice]):
    class DeviceClass:
        @staticmethod
        def defaultDeviceWithDeviceType_mediaType_position_(*args):
            return target

        @staticmethod
        def devicesWithMediaType_(media_type):
            return video_devices if media_type == "video" else []

    return SimpleNamespace(
        AVCaptureDevice=DeviceClass,
        AVCaptureDeviceTypeBuiltInWideAngleCamera="builtin-wide",
        AVCaptureDevicePositionUnspecified=0,
        AVMediaTypeVideo="video",
        AVMediaTypeMuxed="muxed",
    )


def test_builtin_selector_does_not_assume_index_zero() -> None:
    iphone = FakeDevice("iphone", continuity=True)
    macbook = FakeDevice("macbook")
    module = fake_avfoundation(target=macbook, video_devices=[iphone, macbook])

    assert resolve_builtin_camera_index(module) == 1


def test_builtin_selector_never_falls_back_to_continuity_camera() -> None:
    iphone = FakeDevice("iphone", continuity=True)
    module = fake_avfoundation(target=None, video_devices=[iphone])

    with pytest.raises(BuiltInCameraUnavailable, match="will not use Continuity Camera"):
        resolve_builtin_camera_index(module)
