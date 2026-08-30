"""Streamlit interface for trained benchmark models and the comparison table."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

import config
from benchmark.data import load_feature
from benchmark.model_registry import REGISTRY, get_spec
from benchmark.train_model import experiment_dir


st.set_page_config(page_title="LAVA Multi-model Benchmark", page_icon="🎙️", layout="wide")
st.title("LAVA – Audio Deepfake Multi-model Benchmark")


def available_models(seed: int) -> list[str]:
    return [
        key
        for key in REGISTRY
        if (experiment_dir(key, seed) / "model.keras").is_file()
        and (experiment_dir(key, seed) / "threshold.txt").is_file()
    ]


@st.cache_resource
def load_detector(model_key: str, seed: int):
    directory = experiment_dir(model_key, seed)
    model = tf.keras.models.load_model(directory / "model.keras", compile=False)
    threshold = float((directory / "threshold.txt").read_text(encoding="utf-8").strip())
    return model, threshold


seed = st.sidebar.number_input("Seed", min_value=0, value=config.RANDOM_SEED, step=1)
trained = available_models(int(seed))
if not trained:
    st.warning("Chưa có experiment hoàn chỉnh. Hãy train/import model và chạy evaluate trước.")
    st.stop()

selected = st.sidebar.selectbox(
    "Model", trained, format_func=lambda key: REGISTRY[key].display_name
)
spec = get_spec(selected)
model, threshold = load_detector(selected, int(seed))
st.sidebar.caption(spec.variant)
st.sidebar.metric("Threshold P(FAKE)", f"{threshold:.3f}")

comparison = Path(config.OUTPUTS_DIR) / "comparison" / f"model_comparison_seed{int(seed)}.csv"
if comparison.is_file():
    st.subheader("Bảng so sánh đã đo")
    with comparison.open("r", encoding="utf-8-sig", newline="") as handle:
        st.dataframe(list(csv.DictReader(handle)), use_container_width=True, hide_index=True)

st.subheader("Kiểm tra một file âm thanh")
uploaded = st.file_uploader(
    "WAV, FLAC, MP3, OGG hoặc M4A", type=["wav", "flac", "mp3", "ogg", "m4a"]
)
if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getbuffer())
        path = handle.name
    try:
        feature = load_feature(path, spec.input_kind, augment=False, seed=int(seed))
        probability = float(
            model(tf.convert_to_tensor(feature[np.newaxis, ...], dtype=tf.float32), training=False)
            .numpy()
            .reshape(-1)[0]
        )
        prediction = "FAKE" if probability >= threshold else "REAL"
        left, right = st.columns(2)
        left.metric("Kết luận", prediction)
        right.metric("P(FAKE)", f"{probability:.2%}")
        st.progress(float(np.clip(probability, 0, 1)))
    finally:
        Path(path).unlink(missing_ok=True)
