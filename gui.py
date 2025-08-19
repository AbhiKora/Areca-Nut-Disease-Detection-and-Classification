import streamlit as st
import numpy as np
import pickle
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# -------------------
# Load models
# -------------------
def load_model(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".h5", ".keras"]:
        return tf.keras.models.load_model(path), "keras"
    elif ext == ".sav":
        return pickle.load(open(path, "rb")), "pickle"
    else:
        st.error(f"Unsupported model format: {ext}")
        st.stop()

MODEL_PATHS = {
    "CNN": "finalized_model_CNN.keras",
    "RandomForest": "finalized_model_RF.sav",
    "SVM": "finalized_model_SVM.sav",
    "MLP": "finalized_model_MLP.keras",
    "DecisionTree": "finalized_model_DT.sav"
}

models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = load_model(path)
    else:
        st.warning(f"⚠ Model file not found: {path}")

CLASSES = ['diseased', 'mild', 'normal']

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
    ]
    return np.array(features, dtype=np.float32)

# -------------------
# Classification & Metrics
# -------------------
def classify_and_get_confidences(features):
    results = {}
    confidences = {}
    features = features.reshape(1, -1)

    for name, (model, model_type) in models.items():
        if model_type == "keras":
            pred_proba = model.predict(features, verbose=0)
            if pred_proba.ndim > 1:
                confidence = float(np.max(pred_proba))
                pred_class = np.argmax(pred_proba, axis=1)[0]
            else:
                confidence = float(pred_proba[0])
                pred_class = int(confidence > 0.5)
        else:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
            else:
                proba = None
                confidence = 1.0  # Assume full confidence if not available
            pred_class = int(model.predict(features)[0])

        results[name] = CLASSES[pred_class]
        confidences[name] = confidence

    return results, confidences

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="🌰 Arecanut Classifier", layout="centered")
st.title("🌰 Arecanut Multi-Model Classification with Metrics")
st.write("Upload an image to classify it with multiple models (CNN, RF, SVM, MLP, DT).")

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

    if st.button("🚀 Classify and Show Metrics"):
        results, confidences = classify_and_get_confidences(features)

        st.subheader("📊 Predictions & Confidence")
        for model_name in results:
            st.write(f"**{model_name}** → {results[model_name]} ({confidences[model_name]*100:.2f}% confidence)")

        # Plot confidence comparison
        st.subheader("📈 Model Confidence Comparison")
        fig, ax = plt.subplots()
        ax.bar(confidences.keys(), confidences.values(), color='skyblue')
        ax.set_ylabel("Confidence")
        ax.set_ylim([0, 1])
        st.pyplot(fig)
