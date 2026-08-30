"""Single production-model artifact contract for the root implementation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import tensorflow as tf

import config


PRODUCTION_MODEL_NOT_FOUND = "Production model not found. Run: python train.py"


def validate_model_contract(model: tf.keras.Model) -> None:
    """Reject artifacts that do not implement the production detector contract."""
    expected_input = (None, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(
            f"Production model input mismatch: expected {expected_input}, got {model.input_shape}"
        )
    if tuple(model.output_shape) != (None, 1):
        raise ValueError(
            f"Production model output mismatch: expected (None, 1), got {model.output_shape}"
        )

    temporal_wrapper = model.get_layer("time_distributed_mobilenetv3small")
    backbone = temporal_wrapper.layer
    if (
        not isinstance(backbone, tf.keras.Model)
        or backbone.name.lower() != "mobilenetv3small"
        or tuple(backbone.input_shape) != (None, *config.IMAGE_SIZE, config.CHANNELS)
        or backbone.output_shape[-1] != 576
    ):
        raise ValueError("Production model must contain the MobileNetV3Small embedding backbone")

    lstm = model.get_layer("temporal_lstm")
    if not isinstance(lstm, tf.keras.layers.LSTM) or lstm.units != config.LSTM_UNITS:
        raise ValueError(f"Production model must contain LSTM({config.LSTM_UNITS})")
    dense = model.get_layer("classifier_dense")
    if (
        dense.units != config.DENSE_UNITS
        or tf.keras.activations.serialize(dense.activation) != "relu"
    ):
        raise ValueError(
            f"Production classifier must contain Dense({config.DENSE_UNITS}, activation='relu')"
        )
    dropout = model.get_layer("classifier_dropout")
    if (
        not isinstance(dropout, tf.keras.layers.Dropout)
        or abs(dropout.rate - config.DROPOUT_RATE) > 1e-9
    ):
        raise ValueError(f"Production classifier must contain Dropout({config.DROPOUT_RATE})")
    output = model.get_layer("probability_fake")
    if output.units != 1 or tf.keras.activations.serialize(output.activation) != "sigmoid":
        raise ValueError("Production output must be one sigmoid unit representing P(FAKE)")


def load_production_model(*, compile: bool = False) -> tf.keras.Model:
    """Load only the canonical production artifact; never fall back to legacy files."""
    if not os.path.isfile(config.MODEL_PATH):
        raise FileNotFoundError(PRODUCTION_MODEL_NOT_FOUND)
    model = tf.keras.models.load_model(config.MODEL_PATH, compile=compile)
    validate_model_contract(model)
    return model


def save_production_model(model: tf.keras.Model) -> tf.keras.Model:
    """Verify a pending save before atomically replacing the production artifact."""
    validate_model_contract(model)
    pending_path = f"{config.MODEL_PATH}.pending.keras"
    if os.path.exists(pending_path):
        os.remove(pending_path)
    try:
        # The selected checkpoint is loaded with compile=False, so saving it
        # directly avoids fragile positional weight copying in nested models.
        model.save(pending_path, include_optimizer=False)
        verified = tf.keras.models.load_model(pending_path, compile=False)
        validate_model_contract(verified)
        os.replace(pending_path, config.MODEL_PATH)
    finally:
        if os.path.exists(pending_path):
            os.remove(pending_path)
    return load_production_model(compile=False)


def _move_preserving_existing(source: Path, destination_dir: Path) -> str:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        stem, suffix = destination.stem, destination.suffix
        index = 1
        while destination.exists():
            destination = destination_dir / f"{stem}_{index}{suffix}"
            index += 1
    shutil.move(str(source), str(destination))
    return str(destination)


def archive_legacy_artifacts() -> list[str]:
    """Move known stage artifacts only after production verification succeeds."""
    archived: list[str] = []
    for name in (
        "best_model_phase1.keras",
        "best_model_phase2.keras",
        "best_model_phase1.h5",
        "best_model_phase2.h5",
    ):
        source = Path(config.MODELS_DIR) / name
        if source.is_file():
            archived.append(_move_preserving_existing(source, Path(config.LEGACY_MODELS_DIR)))
    for name in ("training_history_phase1.png", "training_history_phase2.png"):
        source = Path(config.PLOTS_DIR) / name
        if source.is_file():
            archived.append(_move_preserving_existing(source, Path(config.LEGACY_TRAINING_DIR)))
    return archived
