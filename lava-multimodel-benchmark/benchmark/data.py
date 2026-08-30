"""Common data adapters for spectrogram and waveform detectors."""

from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np
import tensorflow as tf

import config
from benchmark.protocol import Record
from src.augmentation import apply_augmentation
from src.preprocessing import load_audio, process_audio_data, process_audio_file


SPECTROGRAM = "spectrogram"
WAVEFORM = "waveform"


def _seed(path: str, seed: int) -> int:
    return (zlib.crc32(path.encode("utf-8")) + seed) % (2**32)


def load_feature(path: str, input_kind: str, *, augment: bool, seed: int) -> np.ndarray:
    if input_kind == SPECTROGRAM:
        if not augment:
            return process_audio_file(path).astype(np.float32)
        audio = apply_augmentation(load_audio(path), np.random.default_rng(_seed(path, seed)))
        return process_audio_data(audio).astype(np.float32)
    if input_kind == WAVEFORM:
        audio = load_audio(path)
        if augment:
            audio = apply_augmentation(audio, np.random.default_rng(_seed(path, seed)))
        return audio[:, np.newaxis].astype(np.float32)
    raise ValueError(f"Unsupported input kind: {input_kind}")


def create_dataset(
    records: Sequence[Record],
    *,
    input_kind: str,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = [row.path for row in records]
    labels = [row.label for row in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            min(len(paths), config.SHUFFLE_BUFFER_SIZE),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    expected = (
        (config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS)
        if input_kind == SPECTROGRAM
        else (config.TOTAL_SAMPLES, 1)
    )

    def mapping(path_tensor: tf.Tensor, label_tensor: tf.Tensor):
        feature, label = tf.py_function(
            lambda path, y: (
                load_feature(
                    path.numpy().decode("utf-8"),
                    input_kind,
                    augment=training,
                    seed=seed,
                ),
                np.float32(y.numpy()),
            ),
            inp=[path_tensor, label_tensor],
            Tout=[tf.float32, tf.float32],
        )
        feature.set_shape(expected)
        label.set_shape(())
        return feature, label

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(mapping, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def class_weights(records: Sequence[Record]) -> dict[int, float]:
    labels = np.asarray([row.label for row in records], dtype=np.int32)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training split must contain REAL and FAKE")
    total = float(len(labels))
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}
