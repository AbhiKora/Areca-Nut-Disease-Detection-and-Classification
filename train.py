import os
import numpy as np
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

# -----------------------------
# 1. Dataset Loading & Feature Extraction
# -----------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (64, 64)

classes = ['diseased', 'normal']

def extract_features(image_path):
    img = imread(image_path)
    img_gray = rgb2gray(img)
    img_resized = resize(img_gray, IMG_SIZE, anti_aliasing=True)
    img_rescaled = (img_resized * 255).astype(np.uint8)
    
    glcm = graycomatrix(img_rescaled, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [contrast, dissimilarity, homogeneity, energy, correlation]

features, labels = [], []
class_names = os.listdir(DATASET_DIR)

for label_index, class_name in enumerate(class_names):
    img_files = glob.glob(os.path.join(DATASET_DIR, class_name, "*.jpg"))
    for img_path in img_files:
        try:
            feat = extract_features(img_path)
            features.append(feat)
            labels.append(label_index)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

X = np.array(features)
y = np.array(labels)

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Model Training & Evaluation
# -----------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test, save_path):
    """Train, evaluate, save model, and return metrics + confidences."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(X_test)
    else:
        # For models without predict_proba (like SVC without probability=True)
        confidences = np.zeros((len(y_pred), len(class_names)))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    pickle.dump(model, open(save_path, "wb"))
    return metrics, confidences

# -----------------------------
# Train Traditional Models
# -----------------------------
results = {}
confidences_dict = {}

# Decision Tree
metrics_dt, conf_dt = evaluate_model(
    "Decision Tree",
    DecisionTreeClassifier(random_state=42),
    X_train, y_train, X_test, y_test,
    "finalized_model_DT.sav"
)
results["Decision Tree"] = metrics_dt
confidences_dict["Decision Tree"] = conf_dt

# Build CNN Model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train CNN
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
cnn_model.save("finalized_model_CNN.keras")

# Random Forest
metrics_rf, conf_rf = evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train, X_test, y_test,
    "finalized_model_RF.sav"
)
results["Random Forest"] = metrics_rf
confidences_dict["Random Forest"] = conf_rf

# Support Vector Machine
metrics_svm, conf_svm = evaluate_model(
    "Support Vector Machine",
    SVC(kernel='linear', probability=True, random_state=42),
    X_train, y_train, X_test, y_test,
    "finalized_model_SVM.sav"
)
results["Support Vector Machine"] = metrics_svm
confidences_dict["Support Vector Machine"] = conf_svm

# -----------------------------
# 4. MLP (Tabular-based)
# -----------------------------
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)

mlp_model = Sequential()
mlp_model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
mlp_model.add(Dense(32, activation="relu"))
mlp_model.add(Dense(len(class_names), activation="softmax"))

mlp_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
mlp_model.fit(X_train, y_train_cnn, epochs=50, batch_size=8, verbose=0)

loss, acc = mlp_model.evaluate(X_test, y_test_cnn, verbose=0)
y_pred_mlp = np.argmax(mlp_model.predict(X_test), axis=1)
conf_mlp = mlp_model.predict(X_test)

results["MLP"] = {
    "accuracy": acc,
    "precision": precision_score(y_test, y_pred_mlp, average='weighted', zero_division=0),
    "recall": recall_score(y_test, y_pred_mlp, average='weighted', zero_division=0),
    "f1": f1_score(y_test, y_pred_mlp, average='weighted', zero_division=0)
}
confidences_dict["MLP"] = conf_mlp

mlp_model.save("finalized_model_MLP.keras")

# -----------------------------
# Save results for Streamlit
# -----------------------------
np.save("model_results.npy", results)
np.save("model_confidences.npy", confidences_dict)
np.save("class_names.npy", class_names)

print("✅ Training complete. Metrics and confidences saved.")
