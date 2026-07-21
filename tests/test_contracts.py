from pathlib import Path

import numpy as np
import pytest

from eyeline.config import EyeLineConfig, load_config
from eyeline.contracts import ProcessedFrame


def test_default_config_matches_mvp_contract() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    assert (config.camera.width, config.camera.height, config.camera.fps) == (1280, 720, 30)
    assert config.output.device == "OBS Virtual Camera"


def test_invalid_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="strength"):
        EyeLineConfig(correction=type(EyeLineConfig().correction)(strength=1.1))


def test_processed_frame_remains_bgr() -> None:
    blue_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
    result = ProcessedFrame(blue_bgr, 0.0, False, False, 0.0, 0.1, "no_face")
    assert result.frame_bgr[0, 0].tolist() == [255, 0, 0]
