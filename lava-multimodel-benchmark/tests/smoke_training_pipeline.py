"""Fast structural smoke test for the one-run/one-model training contract."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import tensorflow as tf

import config
from src.artifacts import validate_model_contract
from src.metrics import (
    calibrate_threshold,
    get_lifecycle_checkpoint,
    get_stage_callbacks,
    load_threshold,
    save_threshold,
)
from src.model import (
    build_hybrid_model,
    compile_model,
    freeze_backbone_for_warmup,
    parameter_counts,
    unfreeze_backbone_for_finetuning,
)
from src.utils import plot_training_history


def main() -> None:
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    original_checkpoint = config.TRAINING_CHECKPOINT_PATH
    original_threshold = config.THRESHOLD_PATH
    original_history = config.TRAINING_HISTORY_PATH

    with tempfile.TemporaryDirectory(prefix="lava_training_smoke_") as temporary_dir:
        config.TRAINING_CHECKPOINT_PATH = os.path.join(temporary_dir, "best.keras")
        config.THRESHOLD_PATH = os.path.join(temporary_dir, "threshold.txt")
        config.TRAINING_HISTORY_PATH = os.path.join(temporary_dir, "training_history.png")
        final_path = os.path.join(temporary_dir, "final.keras")

        try:
            model, backbone = build_hybrid_model(weights=None)
            validate_model_contract(model)
            inputs = tf.zeros(
                (1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS),
                dtype=tf.float32,
            )
            labels = tf.constant([0.0], dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(1)

            output = model(inputs, training=False)
            assert tuple(output.shape) == (1, 1)
            assert 0.0 <= float(output.numpy()[0, 0]) <= 1.0

            checkpoint = get_lifecycle_checkpoint()
            freeze_backbone_for_warmup(backbone)
            compile_model(model, config.WARMUP_LR)
            warmup_counts = parameter_counts(model)
            warmup_history = model.fit(
                dataset,
                validation_data=dataset,
                epochs=1,
                callbacks=get_stage_callbacks(checkpoint),
                verbose=0,
            )

            unfreeze_backbone_for_finetuning(backbone)
            compile_model(model, config.FINETUNE_LR)
            finetune_counts = parameter_counts(model)
            trainable_batch_norm = sum(
                1
                for layer in backbone.layers
                if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable
            )
            finetune_history = model.fit(
                dataset,
                validation_data=dataset,
                initial_epoch=1,
                epochs=2,
                callbacks=get_stage_callbacks(checkpoint),
                verbose=0,
            )

            checkpoint_model = tf.keras.models.load_model(
                config.TRAINING_CHECKPOINT_PATH, compile=False
            )
            validate_model_contract(checkpoint_model)
            checkpoint_model.save(final_path)
            reloaded = tf.keras.models.load_model(final_path, compile=False)
            validate_model_contract(reloaded)

            threshold, _ = calibrate_threshold([0, 1], [0.2, 0.8])
            save_threshold(threshold)
            assert np.isclose(load_threshold(), threshold, atol=1e-8)
            history_path = plot_training_history(warmup_history, finetune_history)
            assert os.path.isfile(history_path)

            result = {
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "forward_probability": float(output.numpy()[0, 0]),
                "warmup_trainable": warmup_counts[0],
                "finetune_trainable": finetune_counts[0],
                "trainable_batch_normalization_layers": trainable_batch_norm,
                "checkpoint_save_load": True,
                "final_save_load": True,
                "threshold_save_load": True,
                "combined_history": True,
            }
            if finetune_counts[0] <= warmup_counts[0]:
                raise AssertionError("Fine-tuning did not increase trainable parameters")
            if trainable_batch_norm != 0:
                raise AssertionError("BatchNormalization must remain frozen")
            print(json.dumps(result, indent=2))
        finally:
            config.TRAINING_CHECKPOINT_PATH = original_checkpoint
            config.THRESHOLD_PATH = original_threshold
            config.TRAINING_HISTORY_PATH = original_history


if __name__ == "__main__":
    main()
