"""Train one registered detector and calibrate its validation threshold."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from benchmark.data import class_weights, create_dataset
from benchmark.model_registry import (
    compile_binary,
    freeze_backbone,
    get_spec,
    locate_backbone,
    unfreeze_backbone,
)
from benchmark.protocol import describe, load_manifest, manifest_path, manifest_sha256
from src.metrics import calibrate_threshold


def experiment_dir(model_key: str, seed: int) -> Path:
    return Path(config.OUTPUTS_DIR) / "experiments" / model_key / f"seed_{seed}"


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    labels: list[int] = []
    probabilities: list[float] = []
    for features, batch_labels in dataset:
        probabilities.extend(model(features, training=False).numpy().reshape(-1).tolist())
        labels.extend(batch_labels.numpy().astype(int).reshape(-1).tolist())
    return np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)


def callbacks(checkpoint: Path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=config.LR_REDUCTION_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]


def merge_history(target: dict[str, list[float]], history) -> None:
    for key, values in history.history.items():
        target.setdefault(key, []).extend(float(value) for value in values)


def save_history(history: dict[str, list[float]], directory: Path) -> None:
    (directory / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.get("accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(directory / "training_history.png", dpi=170)
    plt.close(figure)


def main(
    model_key: str,
    seed: int,
    warmup_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    overwrite: bool,
) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    spec = get_spec(model_key)
    directory = experiment_dir(model_key, seed)
    model_path = directory / "model.keras"
    checkpoint = directory / "best_checkpoint.keras"
    warmup_checkpoint = directory / "warmup_best.keras"
    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Experiment already exists: {model_path}. Use --overwrite only when retraining intentionally."
        )
    directory.mkdir(parents=True, exist_ok=True)
    for stale in (checkpoint, warmup_checkpoint, model_path):
        if stale.exists():
            stale.unlink()

    splits = load_manifest(seed=seed)
    print(describe(splits))
    train_dataset = create_dataset(
        splits["train"], input_kind=spec.input_kind, batch_size=batch_size, training=True, seed=seed
    )
    validation_dataset = create_dataset(
        splits["validation"],
        input_kind=spec.input_kind,
        batch_size=batch_size,
        training=False,
        seed=seed,
    )
    weights = class_weights(splits["train"])
    model, backbone = spec.builder()
    history: dict[str, list[float]] = {}

    if backbone is not None and warmup_epochs > 0:
        print(f"\n=== {spec.display_name}: warm-up ===")
        freeze_backbone(backbone)
        compile_binary(model, config.WARMUP_LR)
        warmup = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=warmup_epochs,
            class_weight=weights,
            callbacks=callbacks(warmup_checkpoint),
        )
        merge_history(history, warmup)
        model = tf.keras.models.load_model(warmup_checkpoint, compile=False)
        backbone = locate_backbone(model)

    print(f"\n=== {spec.display_name}: {'fine-tuning' if backbone else 'training'} ===")
    if backbone is not None:
        unfreeze_backbone(backbone, config.FINETUNE_LAYERS)
        learning_rate = config.FINETUNE_LR
    else:
        learning_rate = config.WARMUP_LR
    compile_binary(model, learning_rate)
    start_epoch = len(next(iter(history.values()), []))
    final_epoch = start_epoch + finetune_epochs
    trained = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=start_epoch,
        epochs=final_epoch,
        class_weight=weights,
        callbacks=callbacks(checkpoint),
    )
    merge_history(history, trained)

    candidates = [tf.keras.models.load_model(checkpoint, compile=False)]
    if warmup_checkpoint.exists():
        candidates.append(tf.keras.models.load_model(warmup_checkpoint, compile=False))
    labels = np.asarray([row.label for row in splits["validation"]], dtype=np.float32)
    candidate_losses = []
    for candidate in candidates:
        _, candidate_probabilities = collect_predictions(candidate, validation_dataset)
        clipped = np.clip(candidate_probabilities, 1e-7, 1 - 1e-7)
        loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        candidate_losses.append(float(loss))
    best_model = candidates[int(np.argmin(candidate_losses))]
    pending = directory / "model.pending.keras"
    best_model.save(pending, include_optimizer=False)
    os.replace(pending, model_path)
    labels, probabilities = collect_predictions(best_model, validation_dataset)
    threshold, validation_f1 = calibrate_threshold(labels, probabilities)
    (directory / "threshold.txt").write_text(f"{threshold:.8f}\n", encoding="utf-8")
    save_history(history, directory)

    metadata = {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "implementation_variant": spec.variant,
        "input_kind": spec.input_kind,
        "seed": seed,
        "manifest": str(manifest_path(seed).resolve()),
        "manifest_sha256": manifest_sha256(seed),
        "warmup_epochs_configured": warmup_epochs,
        "finetune_epochs_configured": finetune_epochs,
        "epochs_run": len(next(iter(history.values()), [])),
        "threshold": threshold,
        "threshold_source": "validation FAKE-class F1",
        "validation_f1_at_threshold": validation_f1,
        "parameters": int(best_model.count_params()),
        "input_shape": list(best_model.input_shape),
        "output_shape": list(best_model.output_shape),
        "tensorflow": tf.__version__,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for internal_checkpoint in (checkpoint, warmup_checkpoint):
        if internal_checkpoint.exists():
            internal_checkpoint.unlink()
    print(f"\nSaved model: {model_path.resolve()}")
    print(f"Validation threshold: {threshold:.3f}; F1={validation_f1:.4f}")
    print(f"Next: python -m benchmark.evaluate_model --model {model_key} --seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--warmup-epochs", type=int, default=config.WARMUP_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(
        args.model,
        args.seed,
        args.warmup_epochs,
        args.finetune_epochs,
        args.batch_size,
        args.overwrite,
    )
