"""Create the final comparison table and figures from measured metrics only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from benchmark.model_registry import REGISTRY
from benchmark.train_model import experiment_dir


OUTPUT_DIR = Path(config.OUTPUTS_DIR) / "comparison"


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main(seed: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = set()
    for key, spec in REGISTRY.items():
        path = experiment_dir(key, seed) / "metrics.json"
        if not path.is_file():
            rows.append({"model_key": key, "model": spec.display_name, "status": "PENDING"})
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        hashes.add(metrics["manifest_sha256"])
        rows.append(
            {
                "model_key": key,
                "model": spec.display_name,
                "status": "MEASURED",
                "accuracy": metrics["accuracy"],
                "f1_fake": metrics["f1_fake"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
                "eer": metrics["eer"],
                "parameters": metrics["parameters"],
                "model_size_mib": metrics["model_size_mib"],
                "latency_ms": metrics["latency_model_only"]["mean_seconds"] * 1000,
                "end_to_end_ms": metrics["latency_end_to_end"]["mean_seconds"] * 1000,
                "rtf": metrics["rtf_end_to_end"],
            }
        )
    if len(hashes) > 1:
        raise RuntimeError("Refusing to aggregate results from different split manifests")

    csv_path = OUTPUT_DIR / f"model_comparison_seed{seed}.csv"
    fieldnames = [
        "model_key", "model", "status", "accuracy", "f1_fake", "macro_f1", "roc_auc",
        "eer", "parameters", "model_size_mib", "latency_ms", "end_to_end_ms", "rtf",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# LAVA multi-model comparison",
        "",
        f"Seed: `{seed}`. Only rows marked MEASURED come from saved test metrics.",
        "",
        "| Model | Accuracy ↑ | F1 FAKE ↑ | Macro-F1 ↑ | ROC-AUC ↑ | EER ↓ | Params ↓ | Size MiB ↓ | End-to-end ms ↓ | RTF ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "PENDING":
            markdown.append(f"| {row['model']} | … | … | … | … | … | … | … | … | … |")
        else:
            markdown.append(
                f"| {row['model']} | {percent(row['accuracy'])} | {percent(row['f1_fake'])} | "
                f"{percent(row['macro_f1'])} | {percent(row['roc_auc'])} | {percent(row['eer'])} | "
                f"{row['parameters']:,} | {row['model_size_mib']:.2f} | "
                f"{row['end_to_end_ms']:.2f} | {row['rtf']:.4f} |"
            )
    md_path = OUTPUT_DIR / f"model_comparison_seed{seed}.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    measured = [row for row in rows if row["status"] == "MEASURED"]
    if measured:
        labels = [row["model"] for row in measured]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        x = range(len(measured))
        width = 0.25
        axes[0].bar([i - width for i in x], [100 * row["accuracy"] for row in measured], width, label="Accuracy")
        axes[0].bar(x, [100 * row["f1_fake"] for row in measured], width, label="F1 FAKE")
        axes[0].bar([i + width for i in x], [100 * row["roc_auc"] for row in measured], width, label="ROC-AUC")
        axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
        axes[0].set_ylabel("Percent")
        axes[0].set_title("Detection quality")
        axes[0].legend()
        scatter = axes[1].scatter(
            [row["end_to_end_ms"] for row in measured],
            [100 * row["f1_fake"] for row in measured],
            s=[max(40, min(500, row["model_size_mib"] * 12)) for row in measured],
            c=[100 * row["eer"] for row in measured],
            cmap="viridis_r",
        )
        for row in measured:
            axes[1].annotate(row["model"], (row["end_to_end_ms"], 100 * row["f1_fake"]), fontsize=8)
        axes[1].set_xlabel("End-to-end latency (ms)")
        axes[1].set_ylabel("F1 FAKE (%)")
        axes[1].set_title("Accuracy–efficiency trade-off")
        figure.colorbar(scatter, ax=axes[1], label="EER (%)")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"model_comparison_seed{seed}.png", dpi=180)
        plt.close(figure)

    print(f"CSV: {csv_path.resolve()}")
    print(f"Markdown: {md_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    main(args.seed)
