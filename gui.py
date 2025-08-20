import streamlit as st
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------
# Load models
# -------------------
def load_model(path):
    return tf.keras.models.load_model(path)

MODEL_PATHS = {
    "CNN": "CNN_model.keras",
    "ResNet": "ResNet_model.keras",
    "EfficientNet": "EfficientNet_model.keras",
    "MobileNet": "MobileNet_model.keras",
}

models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = load_model(path)
    else:
        st.warning(f"⚠ Model file not found: {path}")

CLASSES = ["Diseased", "Normal"]

# -------------------
# Image preprocessing
# -------------------
def preprocess_image(image):
    img_resized = cv2.resize(image, (224, 224))  # Resize for CNN/ResNet/EfficientNet/MobileNet
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Ensure 3 channels
    img_norm = img_rgb.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=0)

# -------------------
# Classification
# -------------------
def classify_image(image):
    results = {}
    confidences = {}

    for name, model in models.items():
        preds = model.predict(image, verbose=0)

        if preds.shape[1] == 1:  # Binary (sigmoid)
            confidence = float(preds[0][0])
            pred_class = int(confidence > 0.5)
            confidence = confidence if pred_class == 1 else 1 - confidence
        else:  # Softmax (2 classes)
            pred_class = np.argmax(preds, axis=1)[0]
            confidence = float(np.max(preds))

        results[name] = CLASSES[pred_class]
        confidences[name] = confidence

    return results, confidences

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="🌰 Arecanut Classifier", layout="centered")
st.title("🌰 Arecanut Deep Learning Classifier")
st.write("Upload an image to classify it using CNN, ResNet, EfficientNet, and MobileNet.")

uploaded_file = st.file_uploader("📂 Upload Arecanut Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📷 Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

    input_tensor = preprocess_image(image)

    if st.button("🚀 Classify"):
        results, confidences = classify_image(input_tensor)

        st.subheader("📊 Predictions & Confidence")
        for model_name in results:
            st.write(f"**{model_name}** → {results[model_name]} ({confidences[model_name]*100:.2f}% confidence)")

        # Plot confidence comparison
        st.subheader("📈 Model Confidence Comparison")
        fig, ax = plt.subplots()
        ax.bar(confidences.keys(), confidences.values(), color="skyblue")
        ax.set_ylabel("Confidence")
        ax.set_ylim([0, 1])
        st.pyplot(fig)
