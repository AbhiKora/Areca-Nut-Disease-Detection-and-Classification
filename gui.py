import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = (128, 128)  # must match training size
CLASS_NAMES = ["Healthy", "Diseased"]  # update with your dataset classes

# Paths to saved models
MODEL_PATHS = {
    "CNN": "CNN.keras",
    "MobileNetV2": "MobileNetV2.keras",
    "EfficientNetB0": "EfficientNetB0.keras",
    "ResNet50": "ResNet50.keras",
    # "SVM": "svm_model.sav",
    # "Random Forest": "rf_model.sav",
    # "Decision Tree": "dt_model.sav"
}

# -----------------------
# HELPER FUNCTIONS
# -----------------------
@st.cache_resource
def load_keras_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_sklearn_model(path):
    return joblib.load(path)

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_with_keras(model, img_array):
    preds = model.predict(img_array)
    if preds.shape[1] == 1:  # binary classification
        prob = preds[0][0]
        return {CLASS_NAMES[0]: 1-prob, CLASS_NAMES[1]: prob}
    else:  # multi-class
        return {CLASS_NAMES[i]: preds[0][i] for i in range(len(CLASS_NAMES))}

def predict_with_sklearn(model, img_array):
    # Flatten image for sklearn models
    flat = img_array.reshape(1, -1)
    prob = model.predict_proba(flat)[0]
    return {CLASS_NAMES[i]: prob[i] for i in range(len(CLASS_NAMES))}

# -----------------------
# STREAMLIT APP
# -----------------------
st.title("🌴 Areca Nut X-Ray Classifier")
st.write("Upload an X-ray image and compare predictions from different models.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Preprocess
    img, img_array = preprocess_image(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run predictions
    st.subheader("🔮 Predictions")
    confidences = {}
    for model_name, path in MODEL_PATHS.items():
        try:
            if path.endswith(".keras"):
                model = load_keras_model(path)
                preds = predict_with_keras(model, img_array)
            else:  # sklearn models
                model = load_sklearn_model(path)
                preds = predict_with_sklearn(model, img_array)

            confidences[model_name] = preds
            pred_class = max(preds, key=preds.get)
            st.write(f"**{model_name}** → {pred_class} ({max(preds.values()):.2f})")
        except Exception as e:
            st.warning(f"{model_name} failed: {e}")

    # Plot comparison
    if confidences:
        st.subheader("📊 Confidence Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, preds in confidences.items():
            ax.bar([f"{model_name}-{c}" for c in preds.keys()],
                   list(preds.values()), label=model_name)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Confidence")
        plt.title("Model Predictions Comparison")
        st.pyplot(fig)
