"""Independent test-set evaluation with raw-probability ROC-AUC."""

from __future__ import annotations

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
from src.artifacts import load_production_model
from src.dataset import create_tf_dataset, scan_files, split_dataset
from src.metrics import compute_eer, load_threshold


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    probabilities: list[float] = []
    for features, batch_labels in dataset:
        probabilities.extend(model.predict_on_batch(features).reshape(-1).tolist())
        labels.extend(batch_labels.numpy().astype(int).reshape(-1).tolist())
    return np.asarray(labels), np.asarray(probabilities)


def main() -> None:
    real_files, fake_files = scan_files()
    _, _, test_data = split_dataset(real_files, fake_files)
    test_dataset = create_tf_dataset(*test_data, batch_size=config.BATCH_SIZE, training=False)

    print(f"Loading production model: {config.MODEL_PATH}")
    model = load_production_model(compile=False)
    y_true, probabilities = collect_predictions(model, test_dataset)
    threshold = load_threshold()
    predictions = (probabilities >= threshold).astype(np.int32)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, pos_label=config.FAKE_LABEL, zero_division=0)
    recall = recall_score(y_true, predictions, pos_label=config.FAKE_LABEL, zero_division=0)
    f1 = f1_score(y_true, predictions, pos_label=config.FAKE_LABEL, zero_division=0)
    macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, probabilities)
    eer, eer_threshold = compute_eer(y_true, probabilities)
    matrix = confusion_matrix(y_true, predictions, labels=[config.REAL_LABEL, config.FAKE_LABEL])
    tn, fp, fn, tp = matrix.ravel()

    print(f"Threshold (calibrated on validation): {threshold:.3f}")
    print("\n--- Test Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f} (computed from raw P(FAKE))")
    print(f"EER:       {eer:.4f} (interpolated threshold={eer_threshold:.4f})")
    print("\n--- Confusion Matrix (REAL=negative, FAKE=positive) ---")
    print(f"TN={tn} FP={fp}")
    print(f"FN={fn} TP={tp}")
    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[config.REAL_LABEL, config.FAKE_LABEL],
            target_names=[config.REAL_NAME, config.FAKE_NAME],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
