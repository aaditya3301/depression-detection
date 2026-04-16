import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import json
import tensorflow as tf
import wave
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Depression Detection – Audio Analysis",
    page_icon="🧠",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
MODELS_DIR = BASE_DIR / "models"
TEST_AUDIO_PATH = AUDIO_DIR / "test_audio.wav"
DEFAULT_MODEL_PATH = MODELS_DIR / "depression_model.h5"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# ── Load model config (feature params + threshold) ─────────────────────────────
DEFAULT_CONFIG = {
    "n_mfcc": 40,
    "max_len": 300,
    "sr": 16000,
    "n_features": 120,
    "best_threshold": 0.45,
}

if MODEL_CONFIG_PATH.exists():
    with open(MODEL_CONFIG_PATH) as f:
        MODEL_CONFIG = {**DEFAULT_CONFIG, **json.load(f)}
else:
    MODEL_CONFIG = DEFAULT_CONFIG


def find_default_model_path() -> Path | None:
    """Find a usable model inside models/ with common Keras formats."""
    preferred_names = [
        "depression_model.keras",
        "depression_model.h5",
        "best_depression_model.keras",
        "best_depression_model.h5",
    ]

    for name in preferred_names:
        candidate = MODELS_DIR / name
        if candidate.exists():
            return candidate

    file_candidates = []
    for ext in ("*.keras", "*.h5", "*.hdf5"):
        file_candidates.extend(MODELS_DIR.glob(ext))

    if file_candidates:
        return max(file_candidates, key=lambda p: p.stat().st_mtime)

    saved_model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "saved_model.pb").exists()]
    if saved_model_dirs:
        return max(saved_model_dirs, key=lambda p: p.stat().st_mtime)

    return None


def create_test_audio(file_path: Path, sample_rate: int = 16000, duration_sec: float = 2.0) -> None:
    """Create a small synthetic WAV file for quick app testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = 0.25 * np.sin(2 * np.pi * 220 * t)
    pcm_audio = (tone * 32767).astype(np.int16)

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())


def ensure_project_structure() -> None:
    """Ensure audio/models folders exist and create a default test audio file."""
    AUDIO_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    if not TEST_AUDIO_PATH.exists():
        create_test_audio(TEST_AUDIO_PATH)


ensure_project_structure()

# ── Helpers ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    """Load the trained Keras model (cached across reruns)."""
    return tf.keras.models.load_model(model_path)


def extract_mfcc(file_path: str, max_len: int = None, n_mfcc: int = None) -> np.ndarray | None:
    """Extract MFCC + delta + delta-delta features — mirrors the fixed notebook."""
    if max_len is None:
        max_len = MODEL_CONFIG["max_len"]
    if n_mfcc is None:
        n_mfcc = MODEL_CONFIG["n_mfcc"]
    sr = MODEL_CONFIG["sr"]

    try:
        y, _ = librosa.load(file_path, sr=sr)

        # Remove leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=25)

        # MFCC + delta + delta-delta
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0).T  # (T, n_mfcc*3)

        # Per-sample z-normalization
        mean = features.mean(axis=0, keepdims=True)
        std  = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        # Pad or truncate
        if features.shape[0] < max_len:
            features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)))
        else:
            features = features[:max_len]

        return features.astype(np.float32)
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None


def predict(model, features: np.ndarray) -> tuple[float, str]:
    """Run inference and return (probability, label)."""
    threshold = MODEL_CONFIG["best_threshold"]
    x = features[np.newaxis, ...]          # (1, max_len, n_features)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "Depressed" if prob >= threshold else "Not Depressed"
    return prob, label


# ── UI ──────────────────────────────────────────────────────────────────────────
st.title("Depression Detection from Audio")
st.markdown(
    "Upload a `.wav` audio clip. The model extracts **MFCC features** and runs "
    "an **LSTM classifier** trained on the DAIC-WOZ dataset to screen for "
    "signs of depression."
)
st.caption(f"Project folders ready: {AUDIO_DIR.name}/ and {MODELS_DIR.name}/")
st.divider()

# Model path input
st.subheader("1 · Load model")
auto_model_path = find_default_model_path()
model_path = st.text_input(
    "Path to your saved Keras model (.keras/.h5 or SavedModel folder)",
    value=str(auto_model_path) if auto_model_path else "",
    placeholder="e.g. models/depression_model.keras",
)

model = None
if not model_path:
    st.info(f"Add your trained model in {MODELS_DIR} or type a custom path.")
elif model_path:
    if os.path.exists(model_path):
        with st.spinner("Loading model…"):
            try:
                model = load_model(model_path)
                st.success("✅ Model loaded successfully")
                st.caption(f"Loaded model: {model_path}")
                model.summary(print_fn=lambda x: None)  # warm-up
            except Exception as e:
                st.error(f"Could not load model: {e}")
    else:
        st.warning("⚠️  File / folder not found. Check the path.")

st.divider()

# Audio source
st.subheader("2 · Choose audio")
audio_source = st.radio(
    "Select audio source",
    ["Use test audio from audio/test_audio.wav", "Upload custom WAV"],
)

features = None
audio_label = ""

if audio_source == "Use test audio from audio/test_audio.wav":
    if TEST_AUDIO_PATH.exists():
        with open(TEST_AUDIO_PATH, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")

        with st.spinner("Extracting MFCC features from test audio…"):
            features = extract_mfcc(str(TEST_AUDIO_PATH))
        audio_label = str(TEST_AUDIO_PATH)
    else:
        st.warning("⚠️  Test audio not found in audio/test_audio.wav")
else:
    uploaded = st.file_uploader(
        "Choose a WAV file (mono or stereo, any sample rate — will be resampled to 16 kHz)",
        type=["wav"],
    )

    if uploaded is not None:
        st.audio(uploaded, format="audio/wav")

        # Save to a temp file so librosa can read it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Extracting MFCC features…"):
            features = extract_mfcc(tmp_path)

        os.unlink(tmp_path)
        audio_label = uploaded.name

if features is not None:
    st.markdown(f"**Audio source:** `{audio_label}`")
    st.markdown(f"**MFCC shape:** `{features.shape}` — ready for inference.")

    st.divider()
    st.subheader("3 · Run prediction")

    if model is None:
        st.info("⬆️  Load a model first (step 1).")
    else:
        if st.button("Analyse audio", type="primary", use_container_width=True):
            with st.spinner("Running inference…"):
                prob, label = predict(model, features)

            # ── Result card ──────────────────────────────────────────────
            color = "#c0392b" if label == "Depressed" else "#27ae60"
            st.markdown(
                f"""
                <div style="
                    background:{color}18;
                    border-left: 5px solid {color};
                    border-radius: 8px;
                    padding: 1.2rem 1.5rem;
                    margin-top: 1rem;
                ">
                    <h2 style="color:{color}; margin:0;">{label}</h2>
                    <p style="margin:0.4rem 0 0; font-size:1rem;">
                         Confidence: <strong>{prob:.1%}</strong>
                        &nbsp;(threshold {MODEL_CONFIG['best_threshold']:.2f})
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Raw probability bar
            st.markdown("")
            st.caption("Raw sigmoid output")
            st.progress(prob)
            st.caption(f"{prob:.4f}")

st.divider()
st.caption(
    "⚠️  This tool is for **research purposes only** and is not a clinical diagnostic instrument. "
    "If you or someone you know is struggling, please reach out to a mental health professional."
)