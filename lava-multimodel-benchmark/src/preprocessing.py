"""Shared audio preprocessing for training, evaluation, CLI, and Streamlit."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Union

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

import config


PathLike = Union[str, os.PathLike[str]]


def normalize_audio_length(audio: np.ndarray) -> np.ndarray:
    """Return mono float32 audio with exactly TOTAL_SAMPLES samples."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite samples")
    if len(audio) < config.TOTAL_SAMPLES:
        audio = np.pad(audio, (0, config.TOTAL_SAMPLES - len(audio)))
    else:
        audio = audio[: config.TOTAL_SAMPLES]
    return audio.astype(np.float32, copy=False)


def load_audio(file_path: PathLike) -> np.ndarray:
    """Load, resample to SAMPLE_RATE, convert to mono, and normalize duration."""
    path = os.fspath(file_path)
    try:
        with sf.SoundFile(path) as audio_file:
            source_rate = int(audio_file.samplerate)
            frame_limit = int(round(source_rate * config.AUDIO_DURATION))
            channels = audio_file.read(frame_limit, dtype="float32", always_2d=True)
        audio = channels.mean(axis=1, dtype=np.float32)
        if source_rate != config.SAMPLE_RATE:
            divisor = int(np.gcd(source_rate, config.SAMPLE_RATE))
            audio = resample_poly(
                audio,
                config.SAMPLE_RATE // divisor,
                source_rate // divisor,
            ).astype(np.float32)
    except (RuntimeError, sf.LibsndfileError):
        # Fallback for containers whose libsndfile lacks a compressed codec.
        import librosa

        audio, _ = librosa.load(
            path,
            sr=config.SAMPLE_RATE,
            mono=True,
            duration=config.AUDIO_DURATION,
        )
    return normalize_audio_length(audio)


def segment_audio(audio: np.ndarray) -> np.ndarray:
    """Split audio chronologically into NUM_SEGMENTS without shuffling."""
    normalized = normalize_audio_length(audio)
    return normalized.reshape(config.NUM_SEGMENTS, config.SEGMENT_SAMPLES)


@lru_cache(maxsize=1)
def _mel_filter_bank() -> np.ndarray:
    """Create an HTK-style triangular Mel filter bank."""
    def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(config.FMIN),
        hz_to_mel(config.FMAX),
        config.N_MELS + 2,
    )
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((config.N_FFT + 1) * hz_points / config.SAMPLE_RATE).astype(int)
    bins = np.clip(bins, 0, config.N_FFT // 2)
    filters = np.zeros((config.N_MELS, config.N_FFT // 2 + 1), dtype=np.float32)
    for index in range(config.N_MELS):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[index, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def create_mel_spectrogram_db(audio_segment: np.ndarray) -> np.ndarray:
    """Create a log-Mel spectrogram in dB for one chronological segment."""
    _, _, spectrum = stft(
        np.asarray(audio_segment, dtype=np.float32),
        fs=config.SAMPLE_RATE,
        window="hann",
        nperseg=config.N_FFT,
        noverlap=config.N_FFT - config.HOP_LENGTH,
        nfft=config.N_FFT,
        boundary="zeros",
        padded=True,
    )
    mel_power = _mel_filter_bank() @ (np.abs(spectrum) ** 2)
    mel_power = np.maximum(mel_power, 1e-10)
    mel_db = 10.0 * np.log10(mel_power)
    mel_db -= float(np.max(mel_db))
    return np.maximum(mel_db, -config.TOP_DB).astype(np.float32)


def create_mel_spectrogram_image(audio_segment: np.ndarray) -> np.ndarray:
    """Convert a segment to a 224x224 RGB tensor in MobileNetV3's 0..255 range."""
    mel_db = create_mel_spectrogram_db(audio_segment)
    mel_0_255 = np.clip(
        (mel_db + config.TOP_DB) * (255.0 / config.TOP_DB),
        config.INPUT_VALUE_MIN,
        config.INPUT_VALUE_MAX,
    )
    resized = cv2.resize(mel_0_255, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat(resized[..., np.newaxis], config.CHANNELS, axis=-1)
    return rgb.astype(np.float32, copy=False)


def process_audio_data(audio_data: np.ndarray) -> np.ndarray:
    """Convert raw mono samples into (NUM_SEGMENTS, 224, 224, 3)."""
    segments = segment_audio(audio_data)
    result = np.stack(
        [create_mel_spectrogram_image(segment) for segment in segments],
        axis=0,
    )
    expected = (config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS)
    if result.shape != expected:
        raise RuntimeError(f"Unexpected preprocessing shape {result.shape}; expected {expected}")
    return result


def process_audio_file(file_path: PathLike) -> np.ndarray:
    """Shared file-to-sequence preprocessing pipeline."""
    return process_audio_data(load_audio(file_path))
