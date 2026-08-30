"""Validation callbacks, threshold calibration, and shared detection metrics."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_curve

import config


def get_lifecycle_checkpoint() -> tf.keras.callbacks.ModelCheckpoint:
    """Create the one val-loss checkpoint shared by the complete training run."""
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=config.TRAINING_CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )


def get_stage_callbacks(
    lifecycle_checkpoint: tf.keras.callbacks.ModelCheckpoint,
) -> list[tf.keras.callbacks.Callback]:
    """Use stage-local stopping/LR control without resetting global model selection."""
    return [
        lifecycle_checkpoint,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.LR_REDUCTION_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            mode="min",
            verbose=1,
        ),
    ]


def calibrate_threshold(y_true: Iterable[int], probabilities: Iterable[float]) -> tuple[float, float]:
    labels = np.asarray(list(y_true), dtype=np.int32)
    probs = np.asarray(list(probabilities), dtype=np.float32)
    if labels.size == 0 or labels.size != probs.size:
        raise ValueError("Calibration labels/probabilities are empty or misaligned")
    candidates = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP / 2,
        config.THRESHOLD_SEARCH_STEP,
    )
    scores = np.asarray(
        [f1_score(labels, probs >= threshold, zero_division=0) for threshold in candidates]
    )
    best_score = float(np.max(scores))
    tied_indices = np.flatnonzero(np.isclose(scores, best_score))
    best_index = int(
        tied_indices[
            np.argmin(np.abs(candidates[tied_indices] - config.DEFAULT_THRESHOLD))
        ]
    )
    return float(candidates[best_index]), float(scores[best_index])


def save_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    temporary_path = config.THRESHOLD_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write(f"{threshold:.8f}\n")
    os.replace(temporary_path, config.THRESHOLD_PATH)


def load_threshold() -> float:
    if not os.path.exists(config.THRESHOLD_PATH):
        raise FileNotFoundError("Calibrated threshold not found. Run: python train.py")
    try:
        with open(config.THRESHOLD_PATH, "r", encoding="utf-8") as handle:
            threshold = float(handle.read().strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid threshold artifact: {config.THRESHOLD_PATH}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(f"Threshold outside [0, 1]: {threshold}")
    return threshold


def compute_eer(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[float, float]:
    """Return linearly interpolated EER and its P(FAKE) operating threshold."""
    y_true = np.asarray(list(labels), dtype=np.int32)
    probs = np.asarray(list(probabilities), dtype=np.float32)
    if y_true.size == 0 or y_true.size != probs.size:
        raise ValueError("EER labels/probabilities are empty or misaligned")
    if np.unique(y_true).size != 2:
        raise ValueError("EER requires both REAL and FAKE labels")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, probs, pos_label=config.FAKE_LABEL
    )
    false_negative_rate = 1.0 - true_positive_rate
    difference = false_positive_rate - false_negative_rate
    crossings = np.flatnonzero(np.signbit(difference[:-1]) != np.signbit(difference[1:]))
    if crossings.size == 0:
        index = int(np.argmin(np.abs(difference)))
        eer = (false_positive_rate[index] + false_negative_rate[index]) / 2.0
        return float(eer), float(thresholds[index])

    left = int(crossings[0])
    right = left + 1
    denominator = difference[left] - difference[right]
    weight = 0.0 if denominator == 0 else difference[left] / denominator
    eer = false_positive_rate[left] + weight * (
        false_positive_rate[right] - false_positive_rate[left]
    )
    threshold = thresholds[left] + weight * (thresholds[right] - thresholds[left])
    return float(eer), float(threshold)
