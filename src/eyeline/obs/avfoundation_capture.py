"""Direct, identity-stable AVFoundation capture for the built-in Mac camera."""

from __future__ import annotations

import logging
import queue
from types import TracebackType
from typing import Any

import numpy as np

from eyeline.obs.capture import CameraOpenError, CameraReadError

LOGGER = logging.getLogger(__name__)
_FRAME_DELEGATE_CLASS: Any | None = None


class AVFoundationCapture:
    """Capture BGR frames from an exact built-in AVCaptureDevice object.

    This deliberately bypasses OpenCV's numeric AVFoundation camera indexes.
    Those indexes can change when an iPhone Continuity Camera connects.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        *,
        startup_timeout_seconds: float = 8.0,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if startup_timeout_seconds <= 0:
            raise ValueError("camera startup timeout must be positive")
        self.width = width
        self.height = height
        self.fps = fps
        self.startup_timeout_seconds = startup_timeout_seconds
        self._frames: queue.Queue[np.ndarray | BaseException] = queue.Queue(maxsize=1)
        self._session: Any | None = None
        self._output: Any | None = None
        self._delegate: Any | None = None
        self._dispatch_queue: Any | None = None
        self.device_name: str | None = None
        self.device_unique_id: str | None = None

    @property
    def is_open(self) -> bool:
        return self._session is not None and bool(self._session.isRunning())

    def open(self) -> AVFoundationCapture:
        if self.is_open:
            return self

        import AVFoundation
        import libdispatch
        import Quartz

        device = _select_physical_builtin_camera(AVFoundation)
        self.device_name = str(_property(device, "localizedName"))
        self.device_unique_id = str(_property(device, "uniqueID"))

        session = AVFoundation.AVCaptureSession.alloc().init()
        session.beginConfiguration()
        try:
            preset = _session_preset(AVFoundation, self.width, self.height)
            if not session.canSetSessionPreset_(preset):
                raise CameraOpenError(
                    f"Built-in camera does not support {self.width}x{self.height} capture"
                )
            session.setSessionPreset_(preset)

            device_input, error = (
                AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
            )
            if device_input is None or error is not None:
                raise CameraOpenError(f"Cannot create built-in camera input: {error}")
            if not session.canAddInput_(device_input):
                raise CameraOpenError("Cannot attach the built-in camera to AVFoundation")
            session.addInput_(device_input)

            output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
            output.setAlwaysDiscardsLateVideoFrames_(True)
            output.setVideoSettings_(
                {
                    Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA,
                }
            )
            if not session.canAddOutput_(output):
                raise CameraOpenError("Cannot attach the BGR video output to AVFoundation")
            session.addOutput_(output)

            delegate = _make_frame_delegate(self)
            dispatch_queue = libdispatch.dispatch_queue_create(
                b"com.eyeline.capture.builtin", None
            )
            output.setSampleBufferDelegate_queue_(delegate, dispatch_queue)
        finally:
            session.commitConfiguration()

        self._session = session
        self._output = output
        self._delegate = delegate
        self._dispatch_queue = dispatch_queue
        LOGGER.info(
            "Opening exact physical built-in camera: %s (unique ID %s)",
            self.device_name,
            self.device_unique_id,
        )
        try:
            session.startRunning()
        except Exception as exc:
            self.close()
            raise CameraOpenError("AVFoundation failed to start the built-in camera") from exc
        if not session.isRunning():
            self.close()
            raise CameraOpenError("AVFoundation failed to start the built-in camera")
        try:
            first_frame = self._next_frame()
        except CameraReadError:
            self.close()
            raise
        _replace_latest(self._frames, first_frame)
        return self

    def read(self) -> np.ndarray:
        if not self.is_open:
            raise CameraReadError("camera is not open")
        frame = self._next_frame()
        if frame.shape[:2] != (self.height, self.width):
            import cv2

            frame = cv2.resize(frame, (self.width, self.height))
        return np.ascontiguousarray(frame)

    def _next_frame(self) -> np.ndarray:
        try:
            item = self._frames.get(timeout=self.startup_timeout_seconds)
        except queue.Empty as exc:
            raise CameraReadError(
                f"Built-in camera returned no frame within {self.startup_timeout_seconds:.1f}s"
            ) from exc
        if isinstance(item, BaseException):
            raise CameraReadError("Failed to decode an AVFoundation camera frame") from item
        return item

    def _receive_sample_buffer(self, sample_buffer: Any) -> None:
        try:
            frame = _sample_buffer_to_bgr(sample_buffer)
            _replace_latest(self._frames, frame)
        except BaseException as exc:
            _replace_latest(self._frames, exc)

    def close(self) -> None:
        output, self._output = self._output, None
        session, self._session = self._session, None
        delegate, self._delegate = self._delegate, None
        if output is not None:
            output.setSampleBufferDelegate_queue_(None, None)
        if session is not None and session.isRunning():
            session.stopRunning()
        if delegate is not None:
            delegate._eyeline_owner = None
        self._dispatch_queue = None
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    def __enter__(self) -> AVFoundationCapture:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _select_physical_builtin_camera(avfoundation_module: Any) -> Any:
    device = avfoundation_module.AVCaptureDevice.defaultDeviceWithDeviceType_mediaType_position_(
        avfoundation_module.AVCaptureDeviceTypeBuiltInWideAngleCamera,
        avfoundation_module.AVMediaTypeVideo,
        avfoundation_module.AVCaptureDevicePositionUnspecified,
    )
    if device is None:
        raise CameraOpenError("The physical built-in Mac camera is unavailable")
    if _bool_property(device, "isContinuityCamera"):
        raise CameraOpenError("Refusing to use an iPhone Continuity Camera")
    expected_type = avfoundation_module.AVCaptureDeviceTypeBuiltInWideAngleCamera
    if _property(device, "deviceType") != expected_type:
        raise CameraOpenError("AVFoundation did not return a physical built-in camera")
    return device


def _session_preset(avfoundation_module: Any, width: int, height: int) -> Any:
    presets = {
        (1920, 1080): avfoundation_module.AVCaptureSessionPreset1920x1080,
        (1280, 720): avfoundation_module.AVCaptureSessionPreset1280x720,
    }
    try:
        return presets[(width, height)]
    except KeyError as exc:
        raise CameraOpenError(f"Unsupported built-in camera resolution: {width}x{height}") from exc


def _make_frame_delegate(owner: AVFoundationCapture) -> Any:
    global _FRAME_DELEGATE_CLASS

    import Foundation
    import objc

    if _FRAME_DELEGATE_CLASS is None:
        protocol = objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")

        class EyeLineFrameDelegate(Foundation.NSObject, protocols=[protocol]):
            def captureOutput_didOutputSampleBuffer_fromConnection_(
                self, output: Any, sample_buffer: Any, connection: Any
            ) -> None:
                del output, connection
                owner = self._eyeline_owner
                if owner is not None:
                    with objc.autorelease_pool():
                        owner._receive_sample_buffer(sample_buffer)

        _FRAME_DELEGATE_CLASS = EyeLineFrameDelegate

    delegate = _FRAME_DELEGATE_CLASS.alloc().init()
    delegate._eyeline_owner = owner
    return delegate


def _sample_buffer_to_bgr(sample_buffer: Any) -> np.ndarray:
    import CoreMedia
    import Quartz

    pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
    if pixel_buffer is None:
        raise ValueError("sample buffer has no image buffer")
    flags = Quartz.kCVPixelBufferLock_ReadOnly
    status = Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, flags)
    if status != 0:
        raise ValueError(f"cannot lock pixel buffer: {status}")
    try:
        width = int(Quartz.CVPixelBufferGetWidth(pixel_buffer))
        height = int(Quartz.CVPixelBufferGetHeight(pixel_buffer))
        bytes_per_row = int(Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer))
        base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        if base is None:
            raise ValueError("pixel buffer has no base address")
        raw = np.frombuffer(base.as_buffer(bytes_per_row * height), dtype=np.uint8)
        rows = raw.reshape(height, bytes_per_row)
        bgra = rows[:, : width * 4].reshape(height, width, 4)
        return np.ascontiguousarray(bgra[:, :, :3])
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, flags)


def _replace_latest(target: queue.Queue[Any], item: Any) -> None:
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    try:
        target.put_nowait(item)
    except queue.Full:  # another callback won the race; newest available frame is sufficient
        pass


def _property(obj: Any, name: str) -> Any:
    value = getattr(obj, name)
    return value() if callable(value) else value


def _bool_property(obj: Any, name: str) -> bool:
    value = getattr(obj, name, False)
    return bool(value() if callable(value) else value)
