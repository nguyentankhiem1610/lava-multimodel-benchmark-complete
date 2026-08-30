"""Evaluate one trained detector on the frozen test split and measure CPU cost."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from benchmark.data import create_dataset, load_feature
from benchmark.model_registry import get_spec
from benchmark.protocol import load_manifest, manifest_sha256
from benchmark.train_model import collect_predictions, experiment_dir
from src.metrics import compute_eer


def timed(operation, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, max(0, int(np.ceil(0.95 * len(ordered))) - 1))]
    return {
        "mean_seconds": float(statistics.fmean(samples)),
        "median_seconds": float(statistics.median(samples)),
        "std_seconds": float(statistics.pstdev(samples)),
        "p95_seconds": float(p95),
        "runs": runs,
        "warmup": warmup,
    }


def rss_mib() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024**2)
    except ImportError:
        return None


def main(model_key: str, seed: int, warmup: int, runs: int) -> dict:
    spec = get_spec(model_key)
    directory = experiment_dir(model_key, seed)
    model_path = directory / "model.keras"
    threshold_path = directory / "threshold.txt"
    if not model_path.is_file() or not threshold_path.is_file():
        raise FileNotFoundError(
            f"Missing model/threshold in {directory}. Train or import the model first."
        )

    splits = load_manifest(seed=seed)
    test_dataset = create_dataset(
        splits["test"],
        input_kind=spec.input_kind,
        batch_size=config.BATCH_SIZE,
        training=False,
        seed=seed,
    )
    rss_before = rss_mib()
    model = tf.keras.models.load_model(model_path, compile=False)
    rss_after_load = rss_mib()
    labels, probabilities = collect_predictions(model, test_dataset)
    threshold = float(threshold_path.read_text(encoding="utf-8").strip())
    predictions = (probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    eer, eer_threshold = compute_eer(labels, probabilities)

    representative = splits["test"][0].path
    feature = load_feature(representative, spec.input_kind, augment=False, seed=seed)
    batch = tf.convert_to_tensor(feature[np.newaxis, ...], dtype=tf.float32)
    model_timing = timed(lambda: model(batch, training=False).numpy(), warmup, runs)
    end_to_end = timed(
        lambda: model(
            tf.convert_to_tensor(
                load_feature(representative, spec.input_kind, augment=False, seed=seed)[np.newaxis, ...]
            ),
            training=False,
        ).numpy(),
        min(2, warmup),
        max(3, min(10, runs)),
    )
    accuracy = float(accuracy_score(labels, predictions))
    f1 = float(f1_score(labels, predictions, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    auc = float(roc_auc_score(labels, probabilities))
    size_mib = float(model_path.stat().st_size / 1024**2)
    result = {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "implementation_variant": spec.variant,
        "input_kind": spec.input_kind,
        "seed": seed,
        "manifest_sha256": manifest_sha256(seed),
        "test_count": int(len(labels)),
        "threshold": threshold,
        "accuracy": accuracy,
        "precision_fake": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_fake": f1,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "parameters": int(model.count_params()),
        "model_size_mib": size_mib,
        "latency_model_only": model_timing,
        "latency_end_to_end": end_to_end,
        "rtf_model_only": float(model_timing["mean_seconds"] / config.AUDIO_DURATION),
        "rtf_end_to_end": float(end_to_end["mean_seconds"] / config.AUDIO_DURATION),
        "rss_mib_before_load": rss_before,
        "rss_mib_after_load": rss_after_load,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        },
    }
    (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (directory / "test_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "probability_fake", "prediction"])
        for record, label, probability, prediction in zip(
            splits["test"], labels, probabilities, predictions
        ):
            writer.writerow([record.path, int(label), float(probability), int(prediction)])

    print(f"\n=== {spec.display_name} / test ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 FAKE:  {f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"EER:      {eer:.4f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(labels, predictions, target_names=["REAL", "FAKE"], zero_division=0))
    print(f"Saved: {(directory / 'metrics.json').resolve()}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    main(args.model, args.seed, args.warmup, args.runs)
