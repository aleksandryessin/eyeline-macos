"""Privacy-safe AVFoundation camera selection for macOS."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class BuiltInCameraUnavailable(RuntimeError):
    """Raised instead of silently falling back to an iPhone or external camera."""


def resolve_builtin_camera_index(avfoundation_module: Any | None = None) -> int:
    """Return OpenCV's AVFoundation index for the physical built-in Mac camera.

    OpenCV selects from ``AVCaptureDevice.devicesWithMediaType`` by array index.
    Index zero is not stable when Continuity Camera is available, so first ask
    AVFoundation specifically for ``builtInWideAngleCamera`` and then find that
    device's exact position in the array OpenCV uses.
    """

    if avfoundation_module is None:
        try:
            import AVFoundation as avfoundation_module
        except ImportError as exc:  # pragma: no cover - guarded by locked runtime
            raise BuiltInCameraUnavailable(
                "AVFoundation support is missing; refusing to choose a camera by ambiguous index"
            ) from exc

    device_class = avfoundation_module.AVCaptureDevice
    target = device_class.defaultDeviceWithDeviceType_mediaType_position_(
        avfoundation_module.AVCaptureDeviceTypeBuiltInWideAngleCamera,
        avfoundation_module.AVMediaTypeVideo,
        avfoundation_module.AVCaptureDevicePositionUnspecified,
    )
    if target is None or _bool_property(target, "isContinuityCamera"):
        raise BuiltInCameraUnavailable(
            "The physical built-in Mac camera is unavailable; EyeLine will not use "
            "Continuity Camera automatically"
        )

    devices = list(device_class.devicesWithMediaType_(avfoundation_module.AVMediaTypeVideo))
    muxed_type = getattr(avfoundation_module, "AVMediaTypeMuxed", None)
    if muxed_type is not None:
        devices.extend(device_class.devicesWithMediaType_(muxed_type))

    target_id = str(_property(target, "uniqueID"))
    for index, device in enumerate(devices):
        if str(_property(device, "uniqueID")) == target_id:
            LOGGER.info(
                "Selected physical built-in camera: %s (AVFoundation index %d)",
                _property(device, "localizedName"),
                index,
            )
            return index
    raise BuiltInCameraUnavailable(
        "The built-in Mac camera is not present in OpenCV's AVFoundation device list"
    )


def _property(obj: Any, name: str) -> Any:
    value = getattr(obj, name)
    return value() if callable(value) else value


def _bool_property(obj: Any, name: str) -> bool:
    value = getattr(obj, name, False)
    return bool(value() if callable(value) else value)
