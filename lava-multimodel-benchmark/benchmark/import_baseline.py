"""Import the already-trained MobileNetV3Small-LSTM into the benchmark layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf

import config
from benchmark.data import create_dataset
from benchmark.model_registry import get_spec
from benchmark.protocol import build_manifest, load_manifest, manifest_sha256
from benchmark.train_model import collect_predictions, experiment_dir
from src.metrics import calibrate_threshold


def main(model_source: str, threshold_source: str | None, seed: int, overwrite: bool) -> None:
    source_model = Path(model_source).resolve()
    if not source_model.is_file():
        raise FileNotFoundError(f"Baseline model not found: {source_model}")

    build_manifest(seed=seed)
    spec = get_spec("mobilenetv3small_lstm")
    model = tf.keras.models.load_model(source_model, compile=False)
    expected = (None, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS)
    if tuple(model.input_shape) != expected or tuple(model.output_shape) != (None, 1):
        raise ValueError(f"Baseline contract mismatch: input={model.input_shape}, output={model.output_shape}")

    source_threshold = Path(threshold_source).resolve() if threshold_source else None
    if source_threshold is not None and source_threshold.is_file():
        threshold = float(source_threshold.read_text(encoding="utf-8").strip())
        threshold_origin = str(source_threshold)
    else:
        splits = load_manifest(seed=seed)
        validation_dataset = create_dataset(
            splits["validation"],
            input_kind=spec.input_kind,
            batch_size=config.BATCH_SIZE,
            training=False,
            seed=seed,
        )
        labels, probabilities = collect_predictions(model, validation_dataset)
        threshold, validation_f1 = calibrate_threshold(labels, probabilities)
        threshold_origin = f"calibrated on frozen validation split; F1={validation_f1:.6f}"
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold outside [0,1]: {threshold}")

    destination = experiment_dir(spec.key, seed)
    destination.mkdir(parents=True, exist_ok=True)
    model_target = destination / "model.keras"
    if model_target.exists() and not overwrite:
        raise FileExistsError(f"Baseline already imported: {model_target}")
    shutil.copy2(source_model, model_target)
    (destination / "threshold.txt").write_text(f"{threshold:.8f}\n", encoding="utf-8")
    metadata = {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "implementation_variant": spec.variant,
        "imported_from": str(source_model),
        "seed": seed,
        "manifest_sha256": manifest_sha256(seed),
        "threshold": threshold,
        "threshold_origin": threshold_origin,
        "parameters": int(model.count_params()),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Imported baseline to: {destination.resolve()}")
    print(f"Next: python -m benchmark.evaluate_model --model {spec.key} --seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", default=config.MODEL_PATH)
    parser.add_argument(
        "--threshold-source",
        default=config.THRESHOLD_PATH if Path(config.THRESHOLD_PATH).is_file() else None,
        help="Optional existing threshold; if absent it is calibrated on validation",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.model_source, args.threshold_source, args.seed, args.overwrite)
