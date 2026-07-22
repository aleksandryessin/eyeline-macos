from __future__ import annotations

from eyeline.config import CameraConfig, CorrectionConfig, EyeLineConfig
from eyeline.engine import factory


class StubLandmarks:
    def detect(self, rgb):
        return None


def test_factory_scales_default_focal_length_with_capture_width(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(factory, "create_landmark_backend", lambda *args, **kwargs: StubLandmarks())
    config = EyeLineConfig(camera=CameraConfig(width=1920, height=1080))

    processor = factory.create_frame_processor(config, model_dir=tmp_path)

    assert processor.corrector.focal_length == 975.0


def test_factory_honors_explicit_focal_length(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(factory, "create_landmark_backend", lambda *args, **kwargs: StubLandmarks())
    config = EyeLineConfig(correction=CorrectionConfig(focal_length=812.5))

    processor = factory.create_frame_processor(config, model_dir=tmp_path)

    assert processor.corrector.focal_length == 812.5
