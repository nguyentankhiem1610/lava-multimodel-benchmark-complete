"""One complete warm-up-to-fine-tuning run producing one production model."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import tensorflow as tf

import config
from src.artifacts import archive_legacy_artifacts, save_production_model
from src.dataset import create_tf_dataset, get_class_weights, scan_files, split_dataset
from src.metrics import (
    calibrate_threshold,
    get_lifecycle_checkpoint,
    get_stage_callbacks,
    save_threshold,
)
from src.model import (
    build_hybrid_model,
    compile_model,
    freeze_backbone_for_warmup,
    parameter_counts,
    unfreeze_backbone_for_finetuning,
)
from src.utils import merge_histories, plot_training_history


def set_reproducible_seed() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)


def collect_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[float] = []
    probabilities: list[float] = []
    for features, batch_labels in dataset:
        batch_probabilities = model.predict_on_batch(features).reshape(-1)
        labels.extend(batch_labels.numpy().reshape(-1).tolist())
        probabilities.extend(batch_probabilities.tolist())
    return np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)


def print_parameter_state(label: str, model: tf.keras.Model) -> None:
    trainable, non_trainable = parameter_counts(model)
    print(f"{label}: trainable={trainable:,}; non-trainable={non_trainable:,}")


def load_best_checkpoint() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Load the best full model and recover its nested backbone reference."""
    if not os.path.isfile(config.TRAINING_CHECKPOINT_PATH):
        raise RuntimeError("Training produced no validation checkpoint")
    model = tf.keras.models.load_model(config.TRAINING_CHECKPOINT_PATH, compile=False)
    backbone = model.get_layer("time_distributed_mobilenetv3small").layer
    return model, backbone


def save_metadata(
    *,
    threshold: float,
    threshold_f1: float,
    warmup_epochs_run: int,
    finetune_epochs_run: int,
    best_epoch: int,
    best_val_loss: float,
) -> None:
    metadata = {
        "model_name": "MobileNetV3Small-LSTM",
        "architecture": (
            "6 chronological Mel-spectrogram segments -> "
            "TimeDistributed(MobileNetV3Small, ImageNet, include_top=False, pooling=avg) -> "
            "LSTM(128) -> Dense(64, relu) -> Dropout(0.4) -> Dense(1, sigmoid)"
        ),
        "training_strategy": "warmup_then_finetune",
        "artifact_provenance": "produced by one complete python train.py lifecycle",
        "training_schedule": {
            "warmup_epochs_configured": config.WARMUP_EPOCHS,
            "warmup_epochs_run": warmup_epochs_run,
            "warmup_learning_rate": config.WARMUP_LR,
            "finetune_epochs_configured": config.FINETUNE_EPOCHS,
            "finetune_epochs_run": finetune_epochs_run,
            "finetune_learning_rate": config.FINETUNE_LR,
            "finetune_backbone_layers": config.FINETUNE_LAYERS,
            "batch_normalization_frozen_during_finetuning": True,
        },
        "selection": {
            "monitor": "val_loss",
            "best_epoch_one_based": best_epoch,
            "best_val_loss": best_val_loss,
            "test_set_used_for_selection": False,
        },
        "final_model_path": os.path.relpath(config.MODEL_PATH, config.BASE_DIR),
        "threshold": threshold,
        "threshold_metric": "validation FAKE-class F1",
        "threshold_validation_f1": threshold_f1,
        "threshold_path": os.path.relpath(config.THRESHOLD_PATH, config.BASE_DIR),
        "training_history_path": os.path.relpath(config.TRAINING_HISTORY_PATH, config.BASE_DIR),
        "input_shape": [config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS],
        "input_scale": [config.INPUT_VALUE_MIN, config.INPUT_VALUE_MAX],
        "label_mapping": {config.REAL_NAME: config.REAL_LABEL, config.FAKE_NAME: config.FAKE_LABEL},
        "probability_semantics": "P(FAKE)",
        "random_seed": config.RANDOM_SEED,
    }
    temporary_path = f"{config.MODEL_METADATA_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(temporary_path, config.MODEL_METADATA_PATH)


def main() -> None:
    set_reproducible_seed()
    print("=== MobileNetV3Small-LSTM: one complete training run ===")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Execution device: {'GPU (' + str(len(gpus)) + ')' if gpus else 'CPU'}")

    real_files, fake_files = scan_files()
    if not real_files or not fake_files:
        raise RuntimeError("Place audio files in data/REAL and data/FAKE before training")
    train_data, val_data, test_data = split_dataset(real_files, fake_files)
    print(f"Dataset: REAL={len(real_files)}, FAKE={len(fake_files)}")
    print(f"Splits: train={len(train_data[0])}, val={len(val_data[0])}, test={len(test_data[0])}")
    print("The test split is reserved for evaluate.py and is not used by training or calibration.")

    train_dataset = create_tf_dataset(
        *train_data,
        batch_size=config.BATCH_SIZE,
        training=True,
    )
    val_dataset = create_tf_dataset(
        *val_data,
        batch_size=config.BATCH_SIZE,
        training=False,
    )
    class_weights = get_class_weights(train_data[1])
    print(f"Training-only class weights: {class_weights}")

    if os.path.exists(config.TRAINING_CHECKPOINT_PATH):
        os.remove(config.TRAINING_CHECKPOINT_PATH)
    lifecycle_checkpoint = get_lifecycle_checkpoint()
    model, backbone = build_hybrid_model()
    print(
        f"Tensor flow: input={model.input_shape} -> embeddings="
        f"(B, {config.NUM_SEGMENTS}, {backbone.output_shape[-1]}) -> output={model.output_shape}"
    )

    print("\n--- Warm-up stage (internal) ---")
    freeze_backbone_for_warmup(backbone)
    compile_model(model, config.WARMUP_LR)
    print_parameter_state("Warm-up", model)
    warmup_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.WARMUP_EPOCHS,
        class_weight=class_weights,
        callbacks=get_stage_callbacks(lifecycle_checkpoint),
    )

    # Fine-tuning starts from the best warm-up validation state, while the same
    # checkpoint callback keeps its global best across the entire lifecycle.
    model, backbone = load_best_checkpoint()
    print("\n--- Fine-tuning stage (internal) ---")
    unfreeze_backbone_for_finetuning(backbone)
    compile_model(model, config.FINETUNE_LR)
    print_parameter_state("Fine-tuning", model)
    warmup_epochs_run = len(warmup_history.epoch)
    finetune_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=warmup_epochs_run,
        epochs=warmup_epochs_run + config.FINETUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=get_stage_callbacks(lifecycle_checkpoint),
    )

    model, backbone = load_best_checkpoint()
    production_model = save_production_model(model)
    history_path = plot_training_history(warmup_history, finetune_history)

    y_val, validation_probabilities = collect_predictions(production_model, val_dataset)
    threshold, best_f1 = calibrate_threshold(y_val, validation_probabilities)
    save_threshold(threshold)

    history = merge_histories((warmup_history, finetune_history))
    validation_losses = np.asarray(history["val_loss"], dtype=np.float64)
    best_index = int(np.argmin(validation_losses))
    finetune_epochs_run = len(finetune_history.epoch)
    save_metadata(
        threshold=threshold,
        threshold_f1=best_f1,
        warmup_epochs_run=warmup_epochs_run,
        finetune_epochs_run=finetune_epochs_run,
        best_epoch=best_index + 1,
        best_val_loss=float(validation_losses[best_index]),
    )

    # Legacy artifacts are moved only after model save/load validation, threshold
    # calibration, history generation, and metadata persistence all succeed.
    archived = archive_legacy_artifacts()
    if os.path.exists(config.TRAINING_CHECKPOINT_PATH):
        os.remove(config.TRAINING_CHECKPOINT_PATH)

    print(f"Validation-calibrated threshold={threshold:.3f}; F1={best_f1:.4f}")
    print(f"Final production model: {config.MODEL_PATH}")
    print(f"Threshold: {config.THRESHOLD_PATH}")
    print(f"Training history: {history_path}")
    if archived:
        print("Archived legacy artifacts:")
        for path in archived:
            print(f"  - {path}")
    print("Training complete. Run: python evaluate.py")


if __name__ == "__main__":
    main()
