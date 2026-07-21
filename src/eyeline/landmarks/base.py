"""Shared landmark backend behavior and safe fallback composition."""

from __future__ import annotations

import logging
from collections.abc import Callable

from eyeline.contracts import FaceGeometry, LandmarkBackend, UInt8Frame

LOG = logging.getLogger(__name__)


class LandmarkBackendUnavailable(RuntimeError):
    """Raised when a requested native/model-backed detector cannot be created."""


class NullLandmarkBackend:
    """Detector used when no implementation can start; always fails open."""

    def detect(self, rgb: UInt8Frame) -> FaceGeometry | None:
        return None

    def close(self) -> None:
        return None


class AutoLandmarkBackend:
    """Try MediaPipe first and Vision when it fails or finds no face.

    A transient MediaPipe exception does not poison the whole camera pipeline. Repeated
    failures disable the primary until ``reset_primary`` is called, avoiding an expensive
    exception on every frame while leaving the native Vision implementation available.
    """

    def __init__(
        self,
        primary: LandmarkBackend | None,
        fallback: LandmarkBackend | None,
        *,
        disable_primary_after: int = 3,
    ) -> None:
        if disable_primary_after < 1:
            raise ValueError("disable_primary_after must be positive")
        self.primary = primary
        self.fallback = fallback
        self.disable_primary_after = disable_primary_after
        self.primary_failures = 0
        self.primary_disabled = primary is None
        self.last_backend: str | None = None

    def detect(self, rgb: UInt8Frame) -> FaceGeometry | None:
        if self.primary is not None and not self.primary_disabled:
            try:
                geometry = self.primary.detect(rgb)
                self.primary_failures = 0
                if geometry is not None:
                    self.last_backend = "mediapipe"
                    return geometry
            except Exception:  # the caller must still receive a camera frame
                self.primary_failures += 1
                LOG.exception("MediaPipe landmark detection failed; trying Apple Vision")
                if self.primary_failures >= self.disable_primary_after:
                    self.primary_disabled = True

        if self.fallback is None:
            self.last_backend = None
            return None
        try:
            geometry = self.fallback.detect(rgb)
            self.last_backend = "vision" if geometry is not None else None
            return geometry
        except Exception:  # native framework errors must not interrupt publishing
            LOG.exception("Apple Vision landmark detection failed")
            self.last_backend = None
            return None

    def reset_primary(self) -> None:
        self.primary_failures = 0
        self.primary_disabled = self.primary is None

    def close(self) -> None:
        for backend in (self.primary, self.fallback):
            close: Callable[[], object] | None = getattr(backend, "close", None)
            if close is not None:
                close()
