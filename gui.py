import streamlit as st
import numpy as np
import pickle
import cv2
import os
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops

# -------------------
# Model loader
# -------------------
def load_model(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".h5", ".keras"]:
        st.info(f"Loading Keras model: {path}")
        return tf.keras.models.load_model(path), "keras"
    elif ext == ".sav":
        st.info(f"Loading Pickle model: {path}")
        return pickle.load(open(path, "rb")), "pickle"
    else:
        st.error(f"Unsupported model format: {ext}")
        st.stop()

# -------------------
# Configuration
# -------------------
MODEL_PATHS = {
    "CNN": "finalized_model_CNN.keras",  # Saved in new Keras format
    "RandomForest": "finalized_model_RF.sav",
    "SVM": "finalized_model_SVM.sav"
}

models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = load_model(path)
    else:
        st.warning(f"⚠ Model file not found: {path}")

CLASSES = ['diseased', 'normal']

# -------------------
# Image processing
# -------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def segment_image(image):
    gray = preprocess_image(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_features(image):
    glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    return np.array(features, dtype=np.float32)

def classify_all_models(features):
    results = {}
    features = features.reshape(1, -1)

    for name, (model, model_type) in models.items():
        if model_type == "keras":
            pred = model.predict(features, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0] if pred.ndim > 1 else int(pred[0] > 0.5)
        else:  # pickle model
            pred_class = int(model.predict(features)[0])
        results[name] = CLASSES[pred_class]
    return results

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="🌰 Arecanut Classifier", layout="centered")
st.title("🌰 Arecanut Multi-Model Classification")
st.write("Upload an image of an arecanut to detect if it's **diseased** or **normal** using multiple models.")

uploaded_file = st.file_uploader("📂 Upload Arecanut Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📷 Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

    segmented = segment_image(image)
    st.subheader("🔍 Segmented Image")
    st.image(segmented, channels="GRAY")

    features = extract_features(preprocess_image(image))
    st.write("🧮 **Extracted Features:**", features)

    if st.button("🚀 Classify with All Models"):
        results = classify_all_models(features)

        st.subheader("📊 Model Predictions")
        for model_name, prediction in results.items():
            st.write(f"**{model_name}** → {prediction}")
