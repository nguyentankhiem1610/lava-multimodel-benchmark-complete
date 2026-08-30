"""Generate figures only from the persisted baseline benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "outputs" / "benchmark" / "current_baseline.json"
OUTPUT_DIR = ROOT / "outputs" / "benchmark" / "figures"


def save_confusion_matrix(evidence: dict) -> None:
    values = evidence["test_detection"]["confusion_matrix"]
    matrix = np.asarray([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=16)
    axis.set_xticks([0, 1], ["REAL", "FAKE"])
    axis.set_yticks([0, 1], ["REAL", "FAKE"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Current baseline test confusion matrix (n=10)")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def save_efficiency(evidence: dict) -> None:
    efficiency = evidence["efficiency"]
    labels = ["Model only", "End to end"]
    latency_ms = [
        efficiency["model_only_timing"]["mean_seconds"] * 1000,
        efficiency["end_to_end_timing_including_preprocessing"]["mean_seconds"] * 1000,
    ]
    rtf = [efficiency["model_only_rtf"], efficiency["end_to_end_rtf"]]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(labels, latency_ms, color=["#3b82f6", "#14b8a6"])
    axes[0].set_ylabel("Mean latency (ms)")
    axes[0].set_title("CPU batch-1 latency")
    axes[1].bar(labels, rtf, color=["#3b82f6", "#14b8a6"])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="real-time boundary")
    axes[1].set_ylabel("Real-Time Factor")
    axes[1].set_title("RTF for 3 s input")
    axes[1].legend(fontsize=8)
    figure.suptitle("MobileNetV3Small-LSTM efficiency on the audited Windows CPU")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "efficiency_baseline.png", dpi=180)
    plt.close(figure)


def save_framework() -> None:
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7)
    axis.axis("off")

    boxes = [
        (0.4, 5.6, 2.1, 0.8, "Audio dataset"),
        (3.0, 5.6, 2.4, 0.8, "Standardized\npreprocessing"),
        (6.0, 5.6, 2.4, 0.8, "Detection model registry\n(1 current baseline)"),
        (0.6, 3.5, 2.3, 0.9, "Clean evaluation"),
        (3.85, 3.5, 2.3, 0.9, "Robustness tests"),
        (7.1, 3.5, 2.3, 0.9, "Efficiency tests"),
        (2.0, 1.7, 2.4, 0.8, "Metric aggregation"),
        (5.4, 1.7, 2.1, 0.8, "Pareto analysis"),
        (3.55, 0.25, 3.0, 0.8, "Deployment recommendation"),
    ]
    for x, y, width, height, label in boxes:
        rectangle = plt.Rectangle(
            (x, y), width, height, facecolor="#eff6ff", edgecolor="#1d4ed8", linewidth=1.4
        )
        axis.add_patch(rectangle)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.5, 6.0), (3.0, 6.0)),
        ((5.4, 6.0), (6.0, 6.0)),
        ((7.2, 5.6), (1.75, 4.4)),
        ((7.2, 5.6), (5.0, 4.4)),
        ((7.2, 5.6), (8.25, 4.4)),
        ((1.75, 3.5), (3.2, 2.5)),
        ((5.0, 3.5), (3.2, 2.5)),
        ((8.25, 3.5), (3.2, 2.5)),
        ((4.4, 2.1), (5.4, 2.1)),
        ((6.45, 1.7), (5.05, 1.05)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#475569"})

    axis.text(
        5,
        6.8,
        "LAVA target framework (conceptual; greyed modules are not current results)",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    for index in (4, 7, 8):
        x, y, width, height, _ = boxes[index]
        axis.add_patch(
            plt.Rectangle(
                (x, y), width, height, facecolor="#f3f4f6", edgecolor="#6b7280", linewidth=1.4
            )
        )
        axis.text(x + width / 2, y + height / 2, boxes[index][4], ha="center", va="center", fontsize=10)
    axis.text(
        5,
        3.18,
        "Robustness, multi-model Pareto, and deployment ranking are not implemented",
        ha="center",
        color="#6b7280",
        fontsize=8.5,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "lava_framework.png", dpi=180)
    plt.close(figure)


def main() -> None:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(evidence)
    save_efficiency(evidence)
    save_framework()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
