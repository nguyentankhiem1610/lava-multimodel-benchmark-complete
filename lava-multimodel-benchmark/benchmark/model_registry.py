"""Six-model TensorFlow registry with one P(FAKE) output contract.

RawNet2-LAVA and AASIST-Lite are transparent TensorFlow benchmark
implementations inspired by those architecture families. They are not claimed
to reproduce the authors' official checkpoints or training recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers

import config
from benchmark.data import SPECTROGRAM, WAVEFORM
from src.model import build_hybrid_model


Builder = Callable[[], tuple[tf.keras.Model, tf.keras.Model | None]]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    input_kind: str
    builder: Builder
    variant: str


@tf.keras.utils.register_keras_serializable(package="LAVA")
class ChannelShuffle(layers.Layer):
    def __init__(self, groups: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        channels_per_group = channels // self.groups
        x = tf.reshape(inputs, [batch, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        return tf.reshape(x, [batch, height, width, channels])

    def get_config(self):
        return {**super().get_config(), "groups": self.groups}


def _classifier(inputs, backbone, name: str):
    embeddings = layers.TimeDistributed(backbone, name="time_distributed_backbone")(inputs)
    x = layers.LSTM(config.LSTM_UNITS, name="temporal_lstm")(embeddings)
    x = layers.Dense(config.DENSE_UNITS, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="classifier_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_fake")(x)
    return tf.keras.Model(inputs, outputs, name=name)


def build_mobilenetv3small_lstm():
    return build_hybrid_model(weights="imagenet")


def build_efficientnetb0_lstm():
    inputs = layers.Input(
        (config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS), name="mel_segment_sequence"
    )
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*config.IMAGE_SIZE, config.CHANNELS),
        pooling="avg",
    )
    backbone.trainable = False
    return _classifier(inputs, backbone, "efficientnetb0_lstm_audio_deepfake"), backbone


def _conv_bn_relu(x, filters, kernel=3, stride=1, name="block"):
    x = layers.Conv2D(filters, kernel, stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _shuffle_unit(x, out_channels: int, stride: int, name: str):
    branch_channels = out_channels // 2
    if stride == 1:
        first, second = tf.split(x, 2, axis=-1)
        branch1 = first
    else:
        second = x
        branch1 = layers.DepthwiseConv2D(3, 2, padding="same", use_bias=False, name=f"{name}_b1_dw")(x)
        branch1 = layers.BatchNormalization(name=f"{name}_b1_dw_bn")(branch1)
        branch1 = _conv_bn_relu(branch1, branch_channels, 1, name=f"{name}_b1_pw")
    branch2 = _conv_bn_relu(second, branch_channels, 1, name=f"{name}_b2_pw1")
    branch2 = layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False, name=f"{name}_b2_dw")(branch2)
    branch2 = layers.BatchNormalization(name=f"{name}_b2_dw_bn")(branch2)
    branch2 = _conv_bn_relu(branch2, branch_channels, 1, name=f"{name}_b2_pw2")
    return ChannelShuffle(name=f"{name}_shuffle")(layers.Concatenate()([branch1, branch2]))


def _shufflenet_backbone():
    inputs = layers.Input((*config.IMAGE_SIZE, config.CHANNELS))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = _conv_bn_relu(x, 24, 3, 2, "stem")
    x = layers.MaxPool2D(3, 2, padding="same")(x)
    for stage, (repeats, channels) in enumerate(((4, 116), (8, 232), (4, 464)), start=2):
        x = _shuffle_unit(x, channels, 2, f"stage{stage}_unit1")
        for unit in range(2, repeats + 1):
            x = _shuffle_unit(x, channels, 1, f"stage{stage}_unit{unit}")
    x = _conv_bn_relu(x, 1024, 1, 1, "final")
    outputs = layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, outputs, name="shufflenetv2_1_0")


def build_shufflenetv2_lstm():
    inputs = layers.Input(
        (config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS), name="mel_segment_sequence"
    )
    backbone = _shufflenet_backbone()
    backbone.trainable = True
    return _classifier(inputs, backbone, "shufflenetv2_lstm_audio_deepfake"), backbone


def _inverted_residual(x, out_channels, expansion, stride, kernel, name):
    input_channels = int(x.shape[-1])
    hidden = input_channels * expansion
    residual = x
    if expansion != 1:
        x = _conv_bn_relu(x, hidden, 1, 1, f"{name}_expand")
    x = layers.DepthwiseConv2D(kernel, stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, name=f"{name}_project")(x)
    x = layers.BatchNormalization(name=f"{name}_project_bn")(x)
    if stride == 1 and input_channels == out_channels:
        x = layers.Add()([residual, x])
    return x


def _mnasnet_backbone():
    inputs = layers.Input((*config.IMAGE_SIZE, config.CHANNELS))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = _conv_bn_relu(x, 32, 3, 2, "stem")
    settings = [
        (16, 1, 1, 3, 1),
        (24, 3, 2, 3, 2),
        (40, 3, 2, 5, 3),
        (80, 6, 2, 3, 4),
        (96, 6, 1, 3, 2),
        (192, 6, 2, 5, 3),
        (320, 6, 1, 3, 1),
    ]
    for out_channels, expansion, first_stride, kernel, repeats in settings:
        for repeat in range(repeats):
            x = _inverted_residual(
                x,
                out_channels,
                expansion,
                first_stride if repeat == 0 else 1,
                kernel,
                f"ir_{out_channels}_{repeat + 1}",
            )
    x = _conv_bn_relu(x, 1280, 1, 1, "final")
    return tf.keras.Model(inputs, layers.GlobalAveragePooling2D()(x), name="mnasnet_a1_lava")


def build_mnasnet_lstm():
    inputs = layers.Input(
        (config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS), name="mel_segment_sequence"
    )
    backbone = _mnasnet_backbone()
    return _classifier(inputs, backbone, "mnasnet_lstm_audio_deepfake"), backbone


def _raw_residual(x, filters: int, stride: int, name: str):
    residual = x
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(filters, 3, stride, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    if stride != 1 or int(residual.shape[-1]) != filters:
        residual = layers.Conv1D(filters, 1, stride, padding="same", name=f"{name}_skip")(residual)
    return layers.Add(name=f"{name}_add")([residual, x])


def build_rawnet2_lava():
    inputs = layers.Input((config.TOTAL_SAMPLES, 1), name="raw_waveform")
    x = layers.Conv1D(64, 251, 10, padding="same", use_bias=False, name="raw_frontend")(inputs)
    x = layers.MaxPool1D(3, 3, padding="same")(x)
    for index, (filters, stride) in enumerate(((64, 1), (128, 3), (128, 1), (256, 3), (256, 1))):
        x = _raw_residual(x, filters, stride, f"raw_block{index + 1}")
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.GRU(128, name="temporal_gru")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_fake")(x)
    return tf.keras.Model(inputs, outputs, name="rawnet2_lava_audio_deepfake"), None


def build_aasist_lite():
    inputs = layers.Input((config.TOTAL_SAMPLES, 1), name="raw_waveform")
    x = layers.Conv1D(64, 129, 8, padding="same", use_bias=False, name="spectral_frontend")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    for index, (filters, stride) in enumerate(((64, 2), (128, 2), (128, 2), (192, 2))):
        x = _raw_residual(x, filters, stride, f"encoder_block{index + 1}")
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1, name="graph_attention")(
        x, x
    )
    x = layers.LayerNormalization()(x + attention)
    feed_forward = layers.Dense(256, activation="gelu")(x)
    feed_forward = layers.Dense(192)(feed_forward)
    x = layers.LayerNormalization()(x + feed_forward)
    mean = layers.GlobalAveragePooling1D()(x)
    maximum = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([mean, maximum])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_fake")(x)
    return tf.keras.Model(inputs, outputs, name="aasist_lite_audio_deepfake"), None


REGISTRY: dict[str, ModelSpec] = {
    "mobilenetv3small_lstm": ModelSpec(
        "mobilenetv3small_lstm", "MobileNetV3Small-LSTM", SPECTROGRAM,
        build_mobilenetv3small_lstm, "Keras ImageNet backbone + LSTM(128)",
    ),
    "efficientnetb0_lstm": ModelSpec(
        "efficientnetb0_lstm", "EfficientNet-B0-LSTM", SPECTROGRAM,
        build_efficientnetb0_lstm, "Keras ImageNet backbone + LSTM(128)",
    ),
    "shufflenetv2_lstm": ModelSpec(
        "shufflenetv2_lstm", "ShuffleNetV2-LSTM", SPECTROGRAM,
        build_shufflenetv2_lstm, "LAVA TensorFlow ShuffleNetV2 1.0x + LSTM(128)",
    ),
    "mnasnet_lstm": ModelSpec(
        "mnasnet_lstm", "MnasNet-LSTM", SPECTROGRAM,
        build_mnasnet_lstm, "LAVA TensorFlow MnasNet-A1-style + LSTM(128)",
    ),
    "rawnet2_lava": ModelSpec(
        "rawnet2_lava", "RawNet2-LAVA", WAVEFORM,
        build_rawnet2_lava, "TensorFlow benchmark implementation; not official RawNet2 checkpoint",
    ),
    "aasist_lite": ModelSpec(
        "aasist_lite", "AASIST-Lite", WAVEFORM,
        build_aasist_lite, "TensorFlow lightweight benchmark; not official AASIST checkpoint",
    ),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model {key!r}. Available: {', '.join(REGISTRY)}") from exc


def locate_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, layers.TimeDistributed) and isinstance(layer.layer, tf.keras.Model):
            return layer.layer
    return None


def freeze_backbone(backbone: tf.keras.Model | None) -> None:
    if backbone is not None:
        backbone.trainable = False


def unfreeze_backbone(backbone: tf.keras.Model | None, tail_layers: int) -> None:
    if backbone is None:
        return
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - tail_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_binary(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
