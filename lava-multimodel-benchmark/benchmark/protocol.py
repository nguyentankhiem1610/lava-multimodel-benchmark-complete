"""Freeze and reuse one deterministic train/validation/test protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import config
from src.dataset import scan_files, split_dataset


PROTOCOL_DIR = Path(config.OUTPUTS_DIR) / "protocol"


@dataclass(frozen=True)
class Record:
    split: str
    path: str
    label: int


def manifest_path(seed: int = config.RANDOM_SEED) -> Path:
    return PROTOCOL_DIR / f"splits_seed{seed}.csv"


def _relative(path: str) -> str:
    return Path(path).resolve().relative_to(Path(config.BASE_DIR).resolve()).as_posix()


def build_manifest(*, seed: int = config.RANDOM_SEED, force: bool = False) -> Path:
    """Persist the exact file split. Existing manifests are immutable by default."""
    if seed != config.RANDOM_SEED:
        raise ValueError(
            f"Current split implementation is fixed to RANDOM_SEED={config.RANDOM_SEED}; "
            f"requested {seed}. Change config.RANDOM_SEED before creating another protocol."
        )
    destination = manifest_path(seed)
    if destination.exists() and not force:
        return destination

    real_files, fake_files = scan_files()
    train, validation, test = split_dataset(real_files, fake_files)
    rows: list[Record] = []
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        rows.extend(
            Record(split_name, _relative(path), int(label))
            for path, label in zip(split[0], split[1])
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "path", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"split": row.split, "path": row.path, "label": row.label})
    os.replace(temporary, destination)
    return destination


def load_manifest(*, seed: int = config.RANDOM_SEED) -> dict[str, list[Record]]:
    source = build_manifest(seed=seed)
    result = {"train": [], "validation": [], "test": []}
    with source.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in result:
                raise ValueError(f"Unknown split in manifest: {split}")
            absolute = str((Path(config.BASE_DIR) / row["path"]).resolve())
            if not os.path.isfile(absolute):
                raise FileNotFoundError(f"Manifest audio is missing: {absolute}")
            label = int(row["label"])
            if label not in (config.REAL_LABEL, config.FAKE_LABEL):
                raise ValueError(f"Invalid label {label} for {absolute}")
            result[split].append(Record(split, absolute, label))
    for split, records in result.items():
        if not records:
            raise ValueError(f"Manifest split is empty: {split}")
    return result


def manifest_sha256(seed: int = config.RANDOM_SEED) -> str:
    return hashlib.sha256(manifest_path(seed).read_bytes()).hexdigest()


def describe(splits: dict[str, list[Record]]) -> str:
    lines = []
    for name in ("train", "validation", "test"):
        rows = splits[name]
        real = sum(row.label == config.REAL_LABEL for row in rows)
        fake = sum(row.label == config.FAKE_LABEL for row in rows)
        lines.append(f"{name}: total={len(rows)}, REAL={real}, FAKE={fake}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild the frozen manifest")
    args = parser.parse_args()
    path = build_manifest(force=args.force)
    splits = load_manifest()
    print(f"Manifest: {path.resolve()}")
    print(f"SHA256: {manifest_sha256()}")
    print(describe(splits))
