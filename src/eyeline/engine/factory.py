"""Stable core-engine factory consumed by the OBS runner."""

from __future__ import annotations

import os
from pathlib import Path

from eyeline.config import EyeLineConfig
from eyeline.contracts import FaceGeometry, UInt8Frame
from eyeline.engine.processor import EyeLineFrameProcessor
from eyeline.engine.tensorflow_gaze import TensorFlowGazeCorrector
from eyeline.landmarks.factory import create_landmark_backend
from eyeline.naturalizer import GazeNaturalizer


class LazyTensorFlowGazeCorrector:
    """Delay TensorFlow import/checkpoint allocation until correction is actually needed."""

    def __init__(self, checkpoint_root: Path, *, focal_length: float = 650.0) -> None:
        self.checkpoint_root = checkpoint_root
        self.focal_length = focal_length
        self._corrector: TensorFlowGazeCorrector | None = None
        self._error: Exception | None = None

    def correct(self, frame_bgr: UInt8Frame, geometry: FaceGeometry, strength: float) -> UInt8Frame:
        if self._error is not None:
            raise RuntimeError("gaze model is unavailable") from self._error
        if self._corrector is None:
            try:
                self._corrector = TensorFlowGazeCorrector(
                    self.checkpoint_root,
                    focal_length=self.focal_length,
                )
            except Exception as exc:
                self._error = exc
                raise RuntimeError("gaze model is unavailable") from exc
        return self._corrector.correct(frame_bgr, geometry, strength)

    def close(self) -> None:
        if self._corrector is not None:
            self._corrector.close()


def create_frame_processor(
    config: EyeLineConfig,
    *,
    model_dir: str | Path | None = None,
) -> EyeLineFrameProcessor:
    """Build the configured production processor with all heavy imports lazy/safe."""

    root = _resolve_model_dir(model_dir)
    landmarker = _first_existing(
        root / "face_landmarker.task", root / "models" / "face_landmarker.task"
    )
    landmarks = create_landmark_backend(
        config.landmarks.backend,
        model_path=landmarker,
        min_confidence=config.landmarks.min_confidence,
    )
    backend = config.correction.model_backend.strip().lower()
    if backend not in {"tensorflow", "tf"}:
        raise ValueError(f"unsupported correction backend: {config.correction.model_backend}")
    checkpoint_root = _first_existing_directory(
        root / "weights" / "warping_model" / "flx" / "12",
        root / "warping_model" / "flx" / "12",
        root / "checkpoints",
    )
    # The upstream calibration uses 650 px at 1280-wide capture. Scaling it with
    # the requested width preserves the same field-of-view geometry at 1080p.
    focal_length = config.correction.focal_length
    if focal_length is None:
        focal_length = 650.0 * config.camera.width / 1280.0
    corrector = LazyTensorFlowGazeCorrector(checkpoint_root, focal_length=focal_length)
    naturalizer = GazeNaturalizer(
        config.naturalizer,
        correction_strength=config.correction.strength,
        min_confidence=config.landmarks.min_confidence,
    )
    return EyeLineFrameProcessor(landmarks, corrector, naturalizer)


def _resolve_model_dir(model_dir: str | Path | None) -> Path:
    if model_dir is not None:
        return Path(model_dir).expanduser()
    configured = os.environ.get("EYELINE_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "eyeline" / "models"


def _first_existing(*paths: Path) -> Path:
    return next((path for path in paths if path.is_file()), paths[0])


def _first_existing_directory(*paths: Path) -> Path:
    return next((path for path in paths if path.is_dir()), paths[0])
