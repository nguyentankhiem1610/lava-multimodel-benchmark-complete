"""MobileNetV3Small + LSTM sequence model."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model

import config


def build_hybrid_model(weights: str | None = "imagenet") -> tuple[Model, Model]:
    """Build the required spatial-temporal architecture."""
    inputs = Input(
        shape=(config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS),
        name="mel_segment_sequence",
    )
    backbone = MobileNetV3Small(
        input_shape=(*config.IMAGE_SIZE, config.CHANNELS),
        include_top=False,
        weights=weights,
        pooling="avg",
    )
    backbone.trainable = False

    sequence_embeddings = TimeDistributed(
        backbone,
        name="time_distributed_mobilenetv3small",
    )(inputs)
    temporal_features = LSTM(
        config.LSTM_UNITS,
        return_sequences=False,
        name="temporal_lstm",
    )(sequence_embeddings)
    x = Dense(config.DENSE_UNITS, activation="relu", name="classifier_dense")(temporal_features)
    x = Dropout(config.DROPOUT_RATE, name="classifier_dropout")(x)
    outputs = Dense(1, activation="sigmoid", name="probability_fake")(x)
    model = Model(inputs, outputs, name="mobilenetv3small_lstm_audio_deepfake")
    return model, backbone


def freeze_backbone_for_warmup(backbone: Model) -> None:
    """Freeze the ImageNet backbone while the temporal/classifier head warms up."""
    backbone.trainable = False


def unfreeze_backbone_for_finetuning(backbone: Model) -> None:
    """Unfreeze only the configured tail while keeping BatchNormalization frozen."""
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - config.FINETUNE_LAYERS)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_model(model: Model, learning_rate: float) -> None:
    """Compile after every trainable-state transition."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def parameter_counts(model: Model) -> tuple[int, int]:
    trainable = int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights))
    return trainable, non_trainable
