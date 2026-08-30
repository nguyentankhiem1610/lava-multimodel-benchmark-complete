"""Leakage-resistant file splitting and tf.data input pipelines."""

from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
from src.augmentation import apply_augmentation
from src.preprocessing import load_audio, process_audio_data, process_audio_file


DatasetSplit = tuple[list[str], list[int]]


def _scan_class(directory: str) -> list[str]:
    root = Path(directory)
    return sorted(
        str(path.resolve())
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_AUDIO_EXTENSIONS
    )


def scan_files() -> tuple[list[str], list[str]]:
    """Scan only ROOT/data/REAL and ROOT/data/FAKE."""
    return _scan_class(config.REAL_DIR), _scan_class(config.FAKE_DIR)


def split_dataset(real_files: Sequence[str], fake_files: Sequence[str]) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Split original files before augmentation, stratified by REAL=0/FAKE=1."""
    files = list(real_files) + list(fake_files)
    labels = [config.REAL_LABEL] * len(real_files) + [config.FAKE_LABEL] * len(fake_files)
    if len(real_files) < 3 or len(fake_files) < 3:
        raise ValueError("At least 3 REAL and 3 FAKE files are required for train/val/test")

    holdout_ratio = config.VAL_RATIO + config.TEST_RATIO
    train_files, holdout_files, train_labels, holdout_labels = train_test_split(
        files,
        labels,
        test_size=holdout_ratio,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )
    test_fraction_of_holdout = config.TEST_RATIO / holdout_ratio
    val_files, test_files, val_labels, test_labels = train_test_split(
        holdout_files,
        holdout_labels,
        test_size=test_fraction_of_holdout,
        random_state=config.RANDOM_SEED,
        stratify=holdout_labels,
    )
    return (
        (list(train_files), list(train_labels)),
        (list(val_files), list(val_labels)),
        (list(test_files), list(test_labels)),
    )


def get_class_weights(labels: Sequence[int]) -> dict[int, float]:
    """Compute balanced weights from the training labels only."""
    values = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(values, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training split must contain both REAL and FAKE")
    total = float(len(values))
    return {
        config.REAL_LABEL: total / (2.0 * counts[config.REAL_LABEL]),
        config.FAKE_LABEL: total / (2.0 * counts[config.FAKE_LABEL]),
    }


def _path_seed(path: str) -> int:
    return (zlib.crc32(os.fsencode(path)) + config.RANDOM_SEED) % (2**32)


def _load_example(path_tensor: tf.Tensor, label_tensor: tf.Tensor, augment: bool) -> tuple[np.ndarray, np.float32]:
    path = path_tensor.numpy().decode("utf-8")
    if augment:
        audio = load_audio(path)
        audio = apply_augmentation(audio, np.random.default_rng(_path_seed(path)))
        features = process_audio_data(audio)
    else:
        features = process_audio_file(path)
    return features.astype(np.float32), np.float32(label_tensor.numpy())


def create_tf_dataset(
    file_paths: Sequence[str],
    labels: Sequence[int],
    *,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    """Build a deterministic tf.data pipeline; augmentation is training-only."""
    if len(file_paths) != len(labels):
        raise ValueError("file_paths and labels must have equal length")
    if not file_paths:
        raise ValueError("Cannot create a dataset from an empty split")

    dataset = tf.data.Dataset.from_tensor_slices((list(file_paths), list(labels)))
    if training:
        dataset = dataset.shuffle(
            min(len(file_paths), config.SHUFFLE_BUFFER_SIZE),
            seed=config.RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    def map_example(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        features, output_label = tf.py_function(
            lambda p, y: _load_example(p, y, training),
            inp=[path, label],
            Tout=[tf.float32, tf.float32],
        )
        features.set_shape((config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS))
        output_label.set_shape(())
        return features, output_label

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(map_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)
