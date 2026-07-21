from __future__ import annotations

import numpy as np
from conftest import make_geometry

from eyeline.config import NaturalizerConfig
from eyeline.engine.processor import EyeLineFrameProcessor, bgr_to_rgb
from eyeline.naturalizer import GazeNaturalizer


class Detector:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.received = None

    def detect(self, rgb):
        self.received = rgb.copy()
        if self.error:
            raise self.error
        return self.result


class Corrector:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def correct(self, frame_bgr, geometry, strength):
        self.calls += 1
        if self.error:
            raise self.error
        output = frame_bgr.copy()
        output[..., 1] = 99
        return output

    def close(self):
        return None


def make_processor(detector: Detector, corrector: Corrector) -> EyeLineFrameProcessor:
    config = NaturalizerConfig(fade_in_seconds=0.0, fade_out_seconds=0.0)
    return EyeLineFrameProcessor(
        detector,
        corrector,
        GazeNaturalizer(config, correction_strength=0.8, min_confidence=0.55),
    )


def test_explicit_bgr_to_rgb_conversion_is_packed() -> None:
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)
    assert rgb.flags.c_contiguous
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_processor_sends_rgb_to_landmarks_but_returns_bgr() -> None:
    detector = Detector(make_geometry())
    corrector = Corrector()
    processor = make_processor(detector, corrector)
    source = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    result = processor.process(source, 1.0)
    assert detector.received[0, 0].tolist() == [30, 20, 10]
    assert result.frame_bgr[0, 0].tolist() == [10, 99, 30]
    assert result.correction_applied
    assert source[0, 0].tolist() == [10, 20, 30]


def test_no_face_is_byte_exact_fail_open() -> None:
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    result = make_processor(Detector(None), Corrector()).process(source, 1.0)
    np.testing.assert_array_equal(result.frame_bgr, source)
    assert not result.face_detected
    assert not result.correction_applied
    assert result.reason == "no_face"


def test_landmark_and_model_errors_are_byte_exact_fail_open() -> None:
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    landmark_failure = make_processor(Detector(error=OSError("bad")), Corrector()).process(
        source, 1.0
    )
    np.testing.assert_array_equal(landmark_failure.frame_bgr, source)
    assert landmark_failure.reason == "landmark_error:OSError"

    model_failure = make_processor(
        Detector(make_geometry()), Corrector(error=RuntimeError("bad checkpoint"))
    ).process(source, 1.0)
    np.testing.assert_array_equal(model_failure.frame_bgr, source)
    assert model_failure.face_detected
    assert model_failure.reason == "correction_error:RuntimeError"


def test_rejects_ambiguous_frame_format() -> None:
    processor = make_processor(Detector(None), Corrector())
    with np.testing.assert_raises(ValueError):
        processor.process(np.zeros((4, 4), dtype=np.uint8), 1.0)
