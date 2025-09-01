import json
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_1.h5"
CLASS_NAMES_PATH = BASE_DIR / "cnn" / "class_names.json"
PREPROCESSING_PATH = BASE_DIR / "cnn" / "preprocessing.json"

# === Load Keras model ===
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

# === Load metadata ===
@st.cache_data
def load_metadata():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}")
    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(f"Preprocessing file not found at {PREPROCESSING_PATH}")
    
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    with open(PREPROCESSING_PATH) as f:
        meta = json.load(f)
    return class_names, meta

# === Preprocess image for model ===
def preprocess(img: Image.Image, meta: dict):
    size = tuple(meta.get("img_size", [224, 224]))  # default if missing
    img = img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") * meta.get("rescale", 1.0)
    arr = np.expand_dims(arr, axis=0)
    return arr


# === Make prediction ===
def predict(img: Image.Image):
    model = load_model()
    class_names, meta = load_metadata()
    arr = preprocess(img, meta)
    
    preds = model.predict(arr, verbose=0)[0]  # probabilities per class
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx]), dict(zip(class_names, preds.tolist()))
