"""Train and evaluate registered models sequentially in isolated processes."""

from __future__ import annotations

import argparse
import subprocess
import sys

import config
from benchmark.model_registry import REGISTRY
from benchmark.train_model import experiment_dir


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(models: list[str], seed: int, warmup_epochs: int, finetune_epochs: int) -> None:
    for key in models:
        if key not in REGISTRY:
            raise ValueError(f"Unknown model: {key}")
        model_path = experiment_dir(key, seed) / "model.keras"
        if not model_path.exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.train_model",
                    "--model",
                    key,
                    "--seed",
                    str(seed),
                    "--warmup-epochs",
                    str(warmup_epochs),
                    "--finetune-epochs",
                    str(finetune_epochs),
                ]
            )
        run([sys.executable, "-m", "benchmark.evaluate_model", "--model", key, "--seed", str(seed)])
    run([sys.executable, "-m", "benchmark.aggregate_results", "--seed", str(seed)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(REGISTRY))
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--warmup-epochs", type=int, default=config.WARMUP_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    args = parser.parse_args()
    main(args.models, args.seed, args.warmup_epochs, args.finetune_epochs)
