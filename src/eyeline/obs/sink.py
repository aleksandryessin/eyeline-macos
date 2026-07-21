"""pyvirtualcam output targeting the OBS macOS camera extension."""

from __future__ import annotations

from types import TracebackType
from typing import Any

import numpy as np


class VirtualCameraError(RuntimeError):
    """Raised when the OBS virtual camera cannot be opened or written."""


class OBSVirtualCameraSink:
    """Publish tightly packed RGB frames to OBS Virtual Camera."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        *,
        pyvirtualcam_module: Any | None = None,
        backend: str = "obs",
    ) -> None:
        if pyvirtualcam_module is None:
            import pyvirtualcam as pyvirtualcam_module

        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self._module = pyvirtualcam_module
        self._camera: Any | None = None

    @property
    def device(self) -> str | None:
        return None if self._camera is None else str(self._camera.device)

    def open(self) -> OBSVirtualCameraSink:
        if self._camera is not None:
            return self
        try:
            self._camera = self._module.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=self._module.PixelFormat.RGB,
                backend=self.backend,
            )
        except Exception as exc:
            raise VirtualCameraError(
                "OBS Virtual Camera is unavailable. Close OBS if it is running. Otherwise open "
                "OBS once, choose Start Virtual Camera, then Stop Virtual Camera, quit OBS, "
                "and run EyeLine again."
            ) from exc
        return self

    def send(self, rgb: np.ndarray) -> None:
        if self._camera is None:
            raise VirtualCameraError("OBS Virtual Camera sink is not open")
        expected = (self.height, self.width, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB frame with shape {expected}, got {rgb.shape}")
        try:
            self._camera.send(np.ascontiguousarray(rgb))
            self._camera.sleep_until_next_frame()
        except Exception as exc:
            raise VirtualCameraError("Lost the OBS Virtual Camera output") from exc

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            camera.close()

    def __enter__(self) -> OBSVirtualCameraSink:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
