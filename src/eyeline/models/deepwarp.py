"""Minimal DeepWarp inference graph compatible with the upstream checkpoints.

This is an inference-only adaptation of WangWilly/gaze-correction-cam. Training losses,
summaries, and optimizer state are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelShape:
    height: int = 48
    width: int = 64
    encoded_angle_dim: int = 16


def _conv(tf: Any, inputs: Any, filters: int, kernel: tuple[int, int], name: str) -> Any:
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel,
        padding="same",
        activation=None,
        use_bias=False,
        name=name,
    )(inputs)


def _conv_block(tf: Any, inputs: Any, filters: int, kernel: tuple[int, int], name: str) -> Any:
    with tf.compat.v1.variable_scope(name):
        value = _conv(tf, inputs, filters, kernel, "cnn")
        value = tf.nn.relu(value, name="act")
        return tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            name="bn_layer",
            trainable=True,
        )(value, training=False)


def _dense_block(tf: Any, inputs: Any, units: int, name: str) -> Any:
    with tf.compat.v1.variable_scope(name):
        value = tf.keras.layers.Dense(units=units, activation=None, use_bias=True, name="dnn")(
            inputs
        )
        return tf.nn.relu(value, name="act")


def _transform_module(
    tf: Any,
    inputs: Any,
    depths: tuple[int, ...],
    kernels: tuple[tuple[int, int], ...],
    name: str,
) -> Any:
    with tf.compat.v1.variable_scope(name):
        block_0 = _conv_block(tf, inputs, depths[0], kernels[0], "cnn_blk_0")
        block_1 = _conv_block(tf, block_0, depths[1], kernels[1], "cnn_blk_1")
        block_2 = _conv_block(
            tf, tf.concat([block_0, block_1], axis=3), depths[2], kernels[2], "cnn_blk_2"
        )
        block_3 = _conv_block(
            tf,
            tf.concat([block_0, block_1, block_2], axis=3),
            depths[3],
            kernels[3],
            "cnn_blk_3",
        )
        return _conv(tf, block_3, depths[4], kernels[4], "cnn_4")


def _meshgrid(tf: Any, height: int, width: int) -> Any:
    y_coords = tf.linspace(-1.0, 1.0, height)
    x_coords = tf.linspace(-1.0, 1.0, width)
    x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
    return tf.concat(
        [tf.expand_dims(tf.reshape(x_grid, [-1]), 0), tf.expand_dims(tf.reshape(y_grid, [-1]), 0)],
        axis=0,
    )


def _repeat(tf: Any, vector: Any, repeats: Any) -> Any:
    matrix = tf.matmul(tf.reshape(vector, (-1, 1)), tf.ones((1, repeats), dtype=tf.int32))
    return tf.reshape(matrix, [-1])


def _bilinear(tf: Any, image: Any, sample_x: Any, sample_y: Any, height: int, width: int) -> Any:
    batch_size, image_height, image_width, channels = tf.unstack(tf.shape(image))
    px = 0.5 * (tf.cast(sample_x, tf.float32) + 1.0) * tf.cast(image_width, tf.float32)
    py = 0.5 * (tf.cast(sample_y, tf.float32) + 1.0) * tf.cast(image_height, tf.float32)
    x0, y0 = tf.cast(tf.floor(px), tf.int32), tf.cast(tf.floor(py), tf.int32)
    x1, y1 = x0 + 1, y0 + 1
    x0 = tf.clip_by_value(x0, 0, image_width - 1)
    x1 = tf.clip_by_value(x1, 0, image_width - 1)
    y0 = tf.clip_by_value(y0, 0, image_height - 1)
    y1 = tf.clip_by_value(y1, 0, image_height - 1)
    base = _repeat(tf, tf.range(batch_size) * image_height * image_width, height * width)
    flat = tf.cast(tf.reshape(image, (-1, channels)), tf.float32)
    top_left = tf.gather(flat, base + y0 * image_width + x0)
    bottom_left = tf.gather(flat, base + y1 * image_width + x0)
    top_right = tf.gather(flat, base + y0 * image_width + x1)
    bottom_right = tf.gather(flat, base + y1 * image_width + x1)
    x0f, x1f = tf.cast(x0, tf.float32), tf.cast(x1, tf.float32)
    y0f, y1f = tf.cast(y0, tf.float32), tf.cast(y1, tf.float32)
    weights = (
        tf.expand_dims((x1f - px) * (y1f - py), 1),
        tf.expand_dims((x1f - px) * (py - y0f), 1),
        tf.expand_dims((px - x0f) * (y1f - py), 1),
        tf.expand_dims((px - x0f) * (py - y0f), 1),
    )
    return tf.add_n(
        [
            weights[0] * top_left,
            weights[1] * bottom_left,
            weights[2] * top_right,
            weights[3] * bottom_right,
        ]
    )


def _apply_flow(tf: Any, flow: Any, image: Any, shape: ModelShape) -> Any:
    batch_size = tf.shape(image)[0]
    flow = tf.reshape(tf.transpose(flow, [0, 3, 1, 2]), [batch_size, 2, -1])
    coordinates = flow + _meshgrid(tf, shape.height, shape.width)
    sample_x = tf.reshape(coordinates[:, 0:1, :], [-1])
    sample_y = tf.reshape(coordinates[:, 1:2, :], [-1])
    transformed = _bilinear(tf, image, sample_x, sample_y, shape.height, shape.width)
    return tf.reshape(transformed, [batch_size, shape.height, shape.width, 3])


def build_inference_graph(
    tf: Any, input_image: Any, anchor_points: Any, input_angles: Any, shape: ModelShape
) -> Any:
    """Build the checkpoint-compatible prediction tensor."""

    with tf.compat.v1.variable_scope("warping_model"):
        with tf.compat.v1.variable_scope("encoder"):
            angle = _dense_block(tf, input_angles, 16, "dnn_blk_0")
            angle = _dense_block(tf, angle, 16, "dnn_blk_1")
            angle = _dense_block(tf, angle, shape.encoded_angle_dim, "dnn_blk_2")
            angle = tf.reshape(
                tf.tile(angle, [1, shape.height * shape.width]),
                [-1, shape.height, shape.width, shape.encoded_angle_dim],
            )

        combined = tf.concat([input_image, anchor_points, angle], axis=3)
        with tf.compat.v1.variable_scope("warping_module"):
            coarse = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding="same"
            )(combined)
            coarse = _transform_module(
                tf,
                coarse,
                (32, 64, 64, 32, 16),
                ((5, 5), (3, 3), (3, 3), (3, 3), (1, 1)),
                "coarse_level",
            )
            coarse = tf.nn.tanh(coarse)
            coarse = tf.image.resize(
                coarse,
                (shape.height, shape.width),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            coarse = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(1, 1), padding="same"
            )(coarse)
            fine = _transform_module(
                tf,
                tf.concat([combined, coarse], axis=3),
                (32, 64, 32, 16, 4),
                ((5, 5), (3, 3), (3, 3), (3, 3), (1, 1)),
                "fine_level",
            )
            raw_flow, lcm_input = tf.split(fine, [2, 2], axis=3)

        warped = _apply_flow(tf, tf.nn.tanh(raw_flow), input_image, shape)
        with tf.compat.v1.variable_scope("lcm_module"):
            lcm = _conv_block(tf, lcm_input, 8, (3, 3), "cnn_blk_0")
            lcm = _conv_block(tf, lcm, 8, (3, 3), "cnn_blk_1")
            lcm = tf.nn.softmax(_conv(tf, lcm, 2, (1, 1), "cnn_2"))
        image_weight, palette_weight = tf.split(lcm, [1, 1], axis=3)
        return warped * tf.tile(image_weight, [1, 1, 1, 3]) + tf.ones_like(warped) * tf.tile(
            palette_weight, [1, 1, 1, 3]
        )
