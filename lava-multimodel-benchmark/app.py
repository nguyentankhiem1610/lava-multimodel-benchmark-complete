"""Streamlit interface for the audited root implementation."""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

import config
from src.artifacts import load_production_model
from src.inference import predict_features
from src.metrics import load_threshold
from src.preprocessing import create_mel_spectrogram_db, load_audio, process_audio_file, segment_audio


st.set_page_config(page_title="Audio Deepfake Detection", page_icon="🎙️", layout="wide")


@st.cache_resource
def load_model() -> tuple[tf.keras.Model, float]:
    """Load the one production detector and its validation-calibrated threshold."""
    return load_production_model(compile=False), load_threshold()


def main() -> None:
    st.title("MobileNetV3Small + LSTM Audio Deepfake Detector")
    st.caption("REAL=0, FAKE=1; the model output is P(FAKE).")
    try:
        model, threshold = load_model()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        st.warning(str(exc))
        return

    uploaded = st.file_uploader(
        "Upload audio",
        type=[extension.lstrip(".") for extension in config.SUPPORTED_AUDIO_EXTENSIONS],
    )
    if uploaded is None:
        return
    st.audio(uploaded)

    suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.getvalue())
            temporary_path = handle.name
        audio = load_audio(temporary_path)
        features = process_audio_file(temporary_path)
        result = predict_features(model, features, threshold=threshold)
    except Exception as exc:
        st.error(f"Could not process audio: {exc}")
        return
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    st.subheader("Visualization")
    waveform_column, mel_column = st.columns(2)
    with waveform_column:
        figure, axis = plt.subplots(figsize=(7, 2.5))
        time_axis = [index / config.SAMPLE_RATE for index in range(len(audio))]
        axis.plot(time_axis, audio, linewidth=0.6)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Amplitude")
        axis.set_title("Waveform")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with mel_column:
        first_segment = segment_audio(audio)[0]
        mel_db = create_mel_spectrogram_db(first_segment)
        figure, axis = plt.subplots(figsize=(7, 2.5))
        image = axis.imshow(
            mel_db,
            aspect="auto",
            origin="lower",
            extent=[0, config.SEGMENT_DURATION, 0, config.N_MELS],
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Mel bin")
        axis.set_title("Mel spectrogram - first chronological segment")
        figure.colorbar(image, ax=axis, format="%+2.0f dB")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    color = "red" if result.prediction == config.FAKE_NAME else "green"
    st.markdown(
        f"<h2 style='text-align:center;color:{color}'>Prediction: {result.prediction}</h2>",
        unsafe_allow_html=True,
    )
    first, second, third = st.columns(3)
    first.metric("Confidence", f"{result.confidence * 100:.2f}%")
    second.metric("Raw P(FAKE)", f"{result.probability_fake:.4f}")
    third.metric("Threshold", f"{result.threshold:.4f}")
    st.info(
        f"The LSTM processes {config.NUM_SEGMENTS} ordered segments of "
        f"{config.SEGMENT_DURATION:.1f}s. No unsupported per-segment class predictions are shown."
    )


if __name__ == "__main__":
    main()
