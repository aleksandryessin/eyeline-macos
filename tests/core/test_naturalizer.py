from __future__ import annotations

import numpy as np
import pytest
from conftest import make_geometry

from eyeline.config import NaturalizerConfig
from eyeline.naturalizer import GazeNaturalizer


def naturalizer(**overrides: float) -> GazeNaturalizer:
    values = {
        "smoothing_alpha": 0.5,
        "fade_in_seconds": 0.2,
        "fade_out_seconds": 0.1,
        "head_turn_soft_limit_degrees": 18.0,
        "head_turn_hard_limit_degrees": 32.0,
        "extreme_gaze_threshold": 0.82,
        "blink_threshold": 0.18,
    }
    values.update(overrides)
    return GazeNaturalizer(
        NaturalizerConfig(**values), correction_strength=0.8, min_confidence=0.55
    )


def test_fades_correction_in_and_out_for_blink() -> None:
    subject = naturalizer()
    assert subject.update(make_geometry(), 1.0).effective_strength == 0.0
    assert subject.update(make_geometry(), 1.1).effective_strength == pytest.approx(0.4)
    assert subject.update(make_geometry(), 1.2).effective_strength == pytest.approx(0.8)
    blink = subject.update(make_geometry(openness=0.0), 1.25)
    assert blink.reason is None  # fade is still active while disappearing
    assert blink.effective_strength == pytest.approx(0.4)
    blink = subject.update(make_geometry(openness=0.0), 1.3)
    assert blink.reason == "blink"
    assert blink.effective_strength == 0.0


def test_head_turn_low_confidence_and_extreme_gaze_gate() -> None:
    for geometry, reason in (
        (make_geometry(yaw=35), "head_turn"),
        (make_geometry(confidence=0.4), "low_confidence"),
        (make_geometry(gaze=1.0), "extreme_gaze"),
    ):
        subject = naturalizer(fade_out_seconds=0.0)
        result = subject.update(geometry, 1.0)
        assert result.effective_strength == 0.0
        assert result.reason == reason


def test_moderate_head_turn_weakens_instead_of_switching_abruptly() -> None:
    subject = naturalizer(fade_in_seconds=0.0)
    full = subject.update(make_geometry(), 1.0).effective_strength
    reduced = subject.update(make_geometry(yaw=25.0), 1.1).effective_strength
    assert 0.0 < reduced < full


def test_landmark_jumps_are_ema_smoothed_and_step_limited() -> None:
    subject = naturalizer(fade_in_seconds=0.0)
    first = subject.update(make_geometry(offset=0.0), 1.0).geometry
    second = subject.update(make_geometry(offset=0.3), 1.1).geometry
    assert first is not None and second is not None
    displacement = np.linalg.norm(second.left_eye[0, :2] - first.left_eye[0, :2])
    assert 0.0 < displacement <= 0.035 * 0.5 + 1e-6


def test_missing_face_always_returns_original_contract_strength() -> None:
    subject = naturalizer(fade_in_seconds=0.0)
    subject.update(make_geometry(), 1.0)
    missing = subject.update(None, 1.01)
    assert missing.geometry is None
    assert missing.effective_strength == 0.0
    assert missing.reason == "no_face"
