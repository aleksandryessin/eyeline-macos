from __future__ import annotations

import numpy as np
from conftest import make_geometry

from eyeline.engine.tensorflow_gaze import TensorFlowGazeCorrector, extract_eye_crop


class IdentityEyeModel:
    def __init__(self) -> None:
        self.received = []

    def infer_eye(self, side, image_bgr, anchor_map, angles):
        self.received.append((side, image_bgr.copy(), anchor_map.copy(), angles))
        return image_bgr.copy()

    def close(self):
        return None


def test_eye_extraction_keeps_bgr_channel_order() -> None:
    frame = np.full((120, 160, 3), [10, 20, 240], dtype=np.uint8)
    crop = extract_eye_crop(frame, make_geometry().left_eye, "L")
    assert crop is not None
    np.testing.assert_allclose(crop.image_bgr[24, 32], [10 / 255, 20 / 255, 240 / 255])
    assert crop.anchor_map.shape == (48, 64, 12)


def test_identity_model_composites_without_rgb_swap_or_mutation() -> None:
    model = IdentityEyeModel()
    corrector = TensorFlowGazeCorrector(model=model)
    frame = np.full((120, 160, 3), [10, 20, 240], dtype=np.uint8)
    original = frame.copy()
    output = corrector.correct(frame, make_geometry(), 0.8)
    np.testing.assert_array_equal(frame, original)
    assert len(model.received) == 2
    assert model.received[0][0] == "L"
    np.testing.assert_allclose(model.received[0][1][24, 32], [10 / 255, 20 / 255, 240 / 255])
    assert output[54, 100, 2] > output[54, 100, 0]


def test_crop_near_frame_edge_is_clamped_not_negative_wrapped() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    eye = make_geometry().left_eye.copy()
    eye[:, 0] -= 0.59
    crop = extract_eye_crop(frame, eye, "L")
    assert crop is not None
    assert crop.bounds[2] == 0
