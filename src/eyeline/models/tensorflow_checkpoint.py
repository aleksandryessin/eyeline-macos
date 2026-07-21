"""TensorFlow 2.19 adapter for the upstream compat.v1 checkpoints."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eyeline.models.deepwarp import ModelShape, build_inference_graph


class TensorFlowCheckpointError(RuntimeError):
    """The gaze checkpoint is missing or incompatible."""


class TensorFlowEyeModel:
    """Own independent TF1-style graphs/sessions for left and right eyes."""

    def __init__(self, checkpoint_root: str | Path) -> None:
        root = Path(checkpoint_root)
        self._tf = self._import_tensorflow()
        self._shape = ModelShape()
        self._left = self._load_side(root / "L")
        try:
            self._right = self._load_side(root / "R")
        except Exception:
            self._left[0].close()
            raise
        self._lock = threading.Lock()

    @staticmethod
    def _import_tensorflow() -> Any:
        try:
            import tensorflow as tf
        except Exception as exc:
            raise TensorFlowCheckpointError(f"TensorFlow import failed: {exc}") from exc
        # The release checkpoints use TF1 RefVariable/Saver semantics. This call must happen
        # before building any Keras-backed graph and is intentionally delayed until model use.
        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()
        return tf

    def _load_side(self, directory: Path) -> tuple[Any, Any, Any, Any, Any]:
        tf = self._tf
        checkpoint = tf.train.latest_checkpoint(str(directory))
        if checkpoint is None:
            raise TensorFlowCheckpointError(f"checkpoint not found in {directory}")
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("inputs"):
                image = tf.compat.v1.placeholder(
                    tf.float32, [None, self._shape.height, self._shape.width, 3], name="input_img"
                )
                anchors = tf.compat.v1.placeholder(
                    tf.float32, [None, self._shape.height, self._shape.width, 12], name="input_fp"
                )
                angles = tf.compat.v1.placeholder(tf.float32, [None, 2], name="input_ang")
            prediction = build_inference_graph(tf, image, anchors, angles, self._shape)
            saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            device_count={"GPU": 0},
        )
        session = tf.compat.v1.Session(graph=graph, config=config)
        try:
            saver.restore(session, checkpoint)
        except Exception as exc:
            session.close()
            raise TensorFlowCheckpointError(
                f"failed to restore checkpoint {checkpoint}: {exc}"
            ) from exc
        return session, image, anchors, angles, prediction

    def infer_eye(
        self,
        side: str,
        image_bgr: NDArray[np.float32],
        anchor_map: NDArray[np.float32],
        angles: tuple[float, float],
    ) -> NDArray[np.float32]:
        """Infer one BGR eye crop; no RGB reinterpretation occurs in this path."""

        state = self._left if side.upper() == "L" else self._right
        session, image_tensor, anchor_tensor, angle_tensor, prediction = state
        feed = {
            image_tensor: np.expand_dims(np.asarray(image_bgr, dtype=np.float32), 0),
            anchor_tensor: np.expand_dims(np.asarray(anchor_map, dtype=np.float32), 0),
            angle_tensor: np.asarray([angles], dtype=np.float32),
        }
        with self._lock:
            result = session.run(prediction, feed_dict=feed)
        return np.asarray(result[0], dtype=np.float32)

    def close(self) -> None:
        self._left[0].close()
        self._right[0].close()
