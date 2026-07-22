from pathlib import Path

import numpy as np
import pytest

from eyeline.config import EyeLineConfig, load_config
from eyeline.contracts import ProcessedFrame


def test_default_config_matches_mvp_contract() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    assert (config.camera.width, config.camera.height, config.camera.fps) == (1280, 720, 30)
    assert config.output.device == "OBS Virtual Camera"


def test_zoom_profile_uses_1080p_and_scaled_calibration() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "zoom-1080p.yaml")
    assert (config.camera.width, config.camera.height, config.camera.fps) == (1920, 1080, 30)
    assert config.correction.focal_length == 975.0


def test_invalid_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="strength"):
        EyeLineConfig(correction=type(EyeLineConfig().correction)(strength=1.1))


def test_invalid_focal_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="focal length"):
        EyeLineConfig(correction=type(EyeLineConfig().correction)(focal_length=0.0))


def test_processed_frame_remains_bgr() -> None:
    blue_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
    result = ProcessedFrame(blue_bgr, 0.0, False, False, 0.0, 0.1, "no_face")
    assert result.frame_bgr[0, 0].tolist() == [255, 0, 0]
