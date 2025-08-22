# app.py
import streamlit as st
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="🌰 Arecanut Classifier", layout="centered")

# --- Model Loading ---
@st.cache_resource
def load_tf_model(path):
    """Caches the loaded model to prevent reloading on every interaction."""
    return tf.keras.models.load_model(path)

MODEL_PATHS = {
    "CNN": "CNN_model.keras",
    "ResNet": "ResNet_model.keras",
    "InceptionNet": "InceptionNet_model.keras",
    "MobileNet": "MobileNet_model.keras",
}

models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = load_tf_model(path)
    else:
        st.error(f"⚠️ Model file not found: {path}. Please make sure it's in the same directory.")

CLASSES = ["Diseased", "Mild", "Normal"]

# --- Image Preprocessing ---
def preprocess_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=0)

# --- Classification Logic ---
def classify_image(image):
    results = {}
    confidences = {}

    for name, model in models.items():
        preds = model.predict(image, verbose=0)
        pred_class = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))

        results[name] = CLASSES[pred_class]
        confidences[name] = confidence

    return results, confidences

# --- Streamlit UI ---
st.title("🌰 Arecanut Live Classifier")
st.write("Upload an image to classify it in real-time using four different deep learning models.")

# Add navigation info
st.info("👈 Use the sidebar to navigate to the **Performance Dashboard** to see model evaluation metrics.")

uploaded_file = st.file_uploader("📂 Upload an Arecanut Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📷 Uploaded Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    input_tensor = preprocess_image(image)

    if st.button("🚀 Classify Image"):
        with st.spinner("Classifying..."):
            results, confidences = classify_image(input_tensor)

            st.subheader("📊 Predictions")
            for model_name in results:
                st.write(
                    f"**{model_name}** predicts: **{results[model_name]}** "
                    f"({confidences[model_name]*100:.2f}% confidence)"
                )

            # Plot confidence comparison
            st.subheader("📈 Model Confidence Comparison")
            fig, ax = plt.subplots()
            ax.bar(confidences.keys(), confidences.values(), color="#4CAF50")
            ax.set_ylabel("Confidence Score")
            ax.set_ylim([0, 1])
            plt.xticks(rotation=15)
            st.pyplot(fig)