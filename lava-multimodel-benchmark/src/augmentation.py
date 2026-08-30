"""Training-only waveform augmentation."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample

import config


def apply_augmentation(
    audio: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply one reproducible random augmentation and preserve waveform length."""
    generator = rng or np.random.default_rng(config.RANDOM_SEED)
    result = np.asarray(audio, dtype=np.float32).copy()
    augmentation = generator.choice(("shift", "pitch", "noise", "volume", "none"))

    if augmentation == "shift":
        shift = int(generator.uniform(-0.1, 0.1) * len(result))
        result = np.roll(result, shift)
    elif augmentation == "pitch":
        factor = 2.0 ** (float(generator.uniform(-2.0, 2.0)) / 12.0)
        shifted = resample(result, max(1, int(round(len(result) / factor))))
        result = np.zeros_like(result)
        result[: min(len(result), len(shifted))] = shifted[: len(result)]
    elif augmentation == "noise":
        signal_rms = np.sqrt(np.mean(result**2) + 1e-10)
        snr_db = float(generator.uniform(15.0, 30.0))
        noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        result += generator.normal(0.0, noise_rms, len(result)).astype(np.float32)
    elif augmentation == "volume":
        result *= float(generator.uniform(0.7, 1.3))

    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)
