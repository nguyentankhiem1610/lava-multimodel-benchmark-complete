"""Measure implemented detection and CPU-efficiency metrics for the current baseline.

This runner deliberately does not synthesize multi-model, robustness, or Pareto results.
It evaluates the persisted test split and records a machine-readable evidence artifact.
"""

from __future__ import annotations

import argparse
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.artifacts import load_production_model
from src.dataset import create_tf_dataset, scan_files, split_dataset
from src.metrics import compute_eer, load_threshold
from src.preprocessing import process_audio_file


def collect_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    probabilities: list[float] = []
    for features, batch_labels in dataset:
        probabilities.extend(model(features, training=False).numpy().reshape(-1).tolist())
        labels.extend(batch_labels.numpy().astype(int).reshape(-1).tolist())
    return np.asarray(labels), np.asarray(probabilities)


def timed_runs(operation, warmup: int, runs: int) -> dict[str, float | int]:
    for _ in range(warmup):
        operation()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "warmup_runs": warmup,
        "timed_runs": runs,
        "mean_seconds": float(statistics.fmean(samples)),
        "median_seconds": float(statistics.median(samples)),
        "std_seconds": float(statistics.pstdev(samples)),
        "p95_seconds": float(ordered[p95_index]),
    }


def rss_megabytes() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**2))
    except ImportError:
        if os.name != "nt":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_memory_info.restype = wintypes.BOOL
            process = get_current_process()
            success = get_memory_info(
                process, ctypes.byref(counters), counters.cb
            )
            return float(counters.WorkingSetSize / (1024**2)) if success else None
        except (AttributeError, OSError):
            return None


def main(output_path: str, warmup: int, runs: int, end_to_end_runs: int) -> None:
    real_files, fake_files = scan_files()
    train_data, val_data, test_data = split_dataset(real_files, fake_files)
    test_dataset = create_tf_dataset(
        *test_data, batch_size=config.BATCH_SIZE, training=False
    )

    rss_before_model = rss_megabytes()
    model_path = config.MODEL_PATH
    model = load_production_model(compile=False)
    rss_after_model = rss_megabytes()

    labels, probabilities = collect_predictions(model, test_dataset)
    threshold = load_threshold()
    predictions = (probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(
        labels, predictions, labels=[config.REAL_LABEL, config.FAKE_LABEL]
    )
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    eer, eer_threshold = compute_eer(labels, probabilities)

    representative_path = test_data[0][0]
    features = process_audio_file(representative_path)
    batch = tf.convert_to_tensor(features[np.newaxis, ...], dtype=tf.float32)
    model_timing = timed_runs(
        lambda: model(batch, training=False).numpy(), warmup=warmup, runs=runs
    )
    rss_after_timing = rss_megabytes()
    end_to_end_timing = timed_runs(
        lambda: model(
            tf.convert_to_tensor(
                process_audio_file(representative_path)[np.newaxis, ...], dtype=tf.float32
            ),
            training=False,
        ).numpy(),
        warmup=min(2, warmup),
        runs=end_to_end_runs,
    )
    rss_after_end_to_end = rss_megabytes()

    model_mean = float(model_timing["mean_seconds"])
    end_to_end_mean = float(end_to_end_timing["mean_seconds"])
    observed_rss = [
        value
        for value in (
            rss_before_model,
            rss_after_model,
            rss_after_timing,
            rss_after_end_to_end,
        )
        if value is not None
    ]
    peak_observed_rss = max(observed_rss) if observed_rss else None

    trainable = int(
        sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
    )
    non_trainable = int(
        sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights)
    )
    split_counts = {
        name: {
            "total": len(split[0]),
            "real": int(np.sum(np.asarray(split[1]) == config.REAL_LABEL)),
            "fake": int(np.sum(np.asarray(split[1]) == config.FAKE_LABEL)),
        }
        for name, split in (("train", train_data), ("validation", val_data), ("test", test_data))
    }
    result = {
        "artifact_semantics": "Measured baseline evidence; no unimplemented result is inferred.",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "name": "MobileNetV3Small-LSTM",
            "path": os.path.abspath(model_path),
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "parameters_total": int(model.count_params()),
            "parameters_trainable_at_load": trainable,
            "parameters_non_trainable_at_load": non_trainable,
            "serialized_size_bytes": int(os.path.getsize(model_path)),
            "serialized_size_mib": float(os.path.getsize(model_path) / (1024**2)),
        },
        "dataset": {
            "real": len(real_files),
            "fake": len(fake_files),
            "total": len(real_files) + len(fake_files),
            "splits": split_counts,
            "split_unit": "file",
            "speaker_or_source_grouping": False,
        },
        "test_detection": {
            "threshold": float(threshold),
            "threshold_source": "validation F1 calibration",
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision_fake": float(
                precision_score(labels, predictions, pos_label=config.FAKE_LABEL, zero_division=0)
            ),
            "recall_fake": float(
                recall_score(labels, predictions, pos_label=config.FAKE_LABEL, zero_division=0)
            ),
            "f1_fake": float(
                f1_score(labels, predictions, pos_label=config.FAKE_LABEL, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "eer": eer,
            "eer_threshold": eer_threshold,
            "eer_method": "linear interpolation at the first FPR/FNR crossing",
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        },
        "efficiency": {
            "device": "CPU",
            "batch_size": 1,
            "audio_duration_seconds": config.AUDIO_DURATION,
            "representative_file": os.path.basename(representative_path),
            "model_only_timing": model_timing,
            "model_only_throughput_files_per_second": float(1.0 / model_mean),
            "model_only_rtf": float(model_mean / config.AUDIO_DURATION),
            "end_to_end_timing_including_preprocessing": end_to_end_timing,
            "end_to_end_throughput_files_per_second": float(1.0 / end_to_end_mean),
            "end_to_end_rtf": float(end_to_end_mean / config.AUDIO_DURATION),
            "process_rss_mib": {
                "before_model_load": rss_before_model,
                "after_model_load": rss_after_model,
                "after_model_timing": rss_after_timing,
                "after_end_to_end_timing": rss_after_end_to_end,
                "peak_observed": peak_observed_rss,
                "note": "Process-level RSS snapshots; not isolated model memory or continuous peak sampling.",
            },
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        },
        "not_implemented": [
            "multiple architecture registry",
            "noise stress evaluation",
            "compression stress evaluation",
            "replay stress evaluation",
            "unseen or cross-dataset evaluation",
            "FLOPs or MACs",
            "physical edge-device measurement",
            "multi-model Pareto frontier",
        ],
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved benchmark evidence: {destination.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(config.OUTPUTS_DIR, "benchmark", "current_baseline.json"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--end-to-end-runs", type=int, default=10)
    arguments = parser.parse_args()
    main(arguments.output, arguments.warmup, arguments.runs, arguments.end_to_end_runs)
