"""Plotting helpers for one complete detector-training lifecycle."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config


def merge_histories(histories: Sequence[tf.keras.callbacks.History]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for metric, values in history.history.items():
            merged.setdefault(metric, []).extend(float(value) for value in values)
    return merged


def plot_training_history(
    warmup_history: tf.keras.callbacks.History,
    finetune_history: tf.keras.callbacks.History,
) -> str:
    """Save one plot across warm-up and fine-tuning epochs."""
    history = merge_histories((warmup_history, finetune_history))
    warmup_epochs = len(warmup_history.epoch)
    total_epochs = len(history.get("loss", []))
    epoch_axis = list(range(1, total_epochs + 1))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = (
        (axes[0], "accuracy", "val_accuracy", "Accuracy"),
        (axes[1], "loss", "val_loss", "Binary cross-entropy"),
    )
    for axis, train_key, validation_key, title in panels:
        if train_key in history:
            axis.plot(epoch_axis, history[train_key], label="Train")
        if validation_key in history:
            axis.plot(epoch_axis, history[validation_key], label="Validation")
        if finetune_history.epoch:
            axis.axvline(
                warmup_epochs + 0.5,
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                label="Fine-tuning starts",
            )
        axis.set_xlabel("Lifecycle epoch")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend()

    figure.suptitle("MobileNetV3Small-LSTM complete training lifecycle")
    figure.tight_layout()
    figure.savefig(config.TRAINING_HISTORY_PATH, dpi=160)
    plt.close(figure)
    return config.TRAINING_HISTORY_PATH
