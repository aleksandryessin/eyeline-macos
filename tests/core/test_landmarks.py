from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from conftest import make_geometry

from eyeline.landmarks import AutoLandmarkBackend, MediaPipeTasksLandmarkBackend
from eyeline.landmarks.geometry import head_pose_from_transform


class Backend:
    def __init__(self, results=(), error: Exception | None = None) -> None:
        self.results = list(results)
        self.error = error
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results.pop(0) if self.results else None


def test_auto_does_not_use_vision_when_mediapipe_finds_no_face() -> None:
    expected = make_geometry()
    primary = Backend([None])
    fallback = Backend([expected])
    auto = AutoLandmarkBackend(primary, fallback)
    assert auto.detect(np.zeros((2, 2, 3), dtype=np.uint8)) is None
    assert primary.calls == 1
    assert fallback.calls == 0
    assert auto.last_backend is None


def test_auto_uses_vision_when_mediapipe_raises() -> None:
    expected = make_geometry()
    primary = Backend(error=RuntimeError("MediaPipe failed"))
    fallback = Backend([expected])
    auto = AutoLandmarkBackend(primary, fallback)
    assert auto.detect(np.zeros((2, 2, 3), dtype=np.uint8)) is expected
    assert primary.calls == 1
    assert fallback.calls == 1
    assert auto.last_backend == "vision"


def test_auto_uses_vision_when_primary_is_unavailable() -> None:
    expected = make_geometry()
    fallback = Backend([expected])
    auto = AutoLandmarkBackend(None, fallback)
    assert auto.detect(np.zeros((2, 2, 3), dtype=np.uint8)) is expected
    assert fallback.calls == 1
    assert auto.last_backend == "vision"


def test_auto_disables_repeatedly_crashing_primary() -> None:
    primary = Backend(error=RuntimeError("WindowServer unavailable"))
    fallback = Backend([None, None, None, None])
    auto = AutoLandmarkBackend(primary, fallback, disable_primary_after=3)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    for _ in range(4):
        auto.detect(frame)
    assert primary.calls == 3
    assert fallback.calls == 4
    assert auto.primary_disabled


class FakeImage:
    def __init__(self, *, image_format, data) -> None:
        self.image_format = image_format
        self.data = data.copy()


class FakeLandmarker:
    def __init__(self, result) -> None:
        self.result = result
        self.received = None

    def detect(self, image):
        self.received = image
        return self.result


def point(x=0.5, y=0.5, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z, presence=0.0, visibility=0.0)


def test_mediapipe_tasks_is_given_true_rgb_and_maps_geometry() -> None:
    landmarks = [point() for _ in range(478)]
    # Establish a non-degenerate face and both canonical eye contours.
    landmarks[234], landmarks[454] = point(0.25, 0.5), point(0.75, 0.5)
    landmarks[10], landmarks[152], landmarks[1] = (
        point(0.5, 0.2),
        point(0.5, 0.8),
        point(0.5, 0.5),
    )
    from eyeline.landmarks.mediapipe_tasks import LEFT_EYE_INDICES, RIGHT_EYE_INDICES

    eye_shape = ((0.0, 0.0), (0.02, -0.01), (0.04, -0.01), (0.06, 0), (0.04, 0.01), (0.02, 0.01))
    for indices, base in ((LEFT_EYE_INDICES, 0.62), (RIGHT_EYE_INDICES, 0.32)):
        for index, (dx, dy) in zip(indices, eye_shape, strict=True):
            landmarks[index] = point(base + dx, 0.45 + dy)
    result = SimpleNamespace(face_landmarks=[landmarks], face_blendshapes=[])
    fake_landmarker = FakeLandmarker(result)
    fake_mp = SimpleNamespace(Image=FakeImage, ImageFormat=SimpleNamespace(SRGB="srgb"))
    backend = MediaPipeTasksLandmarkBackend(
        "unused.task", mp_module=fake_mp, landmarker=fake_landmarker
    )
    rgb = np.array([[[250, 10, 4]]], dtype=np.uint8)
    geometry = backend.detect(rgb)
    assert fake_landmarker.received.data[0, 0].tolist() == [250, 10, 4]
    assert geometry is not None
    assert geometry.metadata["backend"] == "mediapipe_tasks"
    assert geometry.left_eye.shape == (6, 3)
    assert geometry.confidence == 1.0


def test_mediapipe_uses_facial_transform_for_calibrated_head_pose() -> None:
    pitch = np.radians(12.0)
    yaw = np.radians(-7.0)
    roll = np.radians(4.0)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x

    pose = head_pose_from_transform(transform)

    assert pose is not None
    np.testing.assert_allclose(pose, (-7.0, 12.0, 4.0), atol=1e-5)
