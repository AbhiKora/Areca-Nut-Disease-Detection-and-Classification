import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from skimage.feature.texture import graycomatrix, graycoprops  # ✅ Updated import
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import glob

# -----------------------------
# 1. Dataset Loading & Feature Extraction
# -----------------------------
DATASET_DIR = "dataset"  # Folder structure: dataset/class_name/*.jpg
IMG_SIZE = (64, 64)       # Resize for consistency

def extract_features(image_path):
    """Extract GLCM features from an image."""
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

features = []
labels = []
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. Model Training Functions
# -----------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test, save_path):
    """Train, evaluate, and save a model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    pickle.dump(model, open(save_path, "wb"))
    print(f"Saved {name} model to {save_path}")

# -----------------------------
# 4. Decision Tree
# -----------------------------
dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model("Decision Tree", dt_model, X_train, y_train, X_test, y_test, "finalized_model_DT.sav")

# -----------------------------
# 5. Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model("Random Forest", rf_model, X_train, y_train, X_test, y_test, "finalized_model_RF.sav")

# -----------------------------
# 6. Support Vector Machine
# -----------------------------
svm_model = SVC(kernel='linear', probability=True, random_state=42)
evaluate_model("Support Vector Machine", svm_model, X_train, y_train, X_test, y_test, "finalized_model_SVM.sav")

# -----------------------------
# 7. CNN/MLP for Tabular Data
# -----------------------------
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)

cnn_model = Sequential()
cnn_model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
cnn_model.add(Dense(32, activation="relu"))
cnn_model.add(Dense(len(class_names), activation="softmax"))

cnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
cnn_model.fit(X_train, y_train_cnn, epochs=50, batch_size=8, verbose=1)

loss, acc = cnn_model.evaluate(X_test, y_test_cnn, verbose=0)
print(f"\nCNN Accuracy: {acc:.4f}")

# keras.saving.save_model(cnn_model, "finalized_model_CNN.keras")
cnn_model.save("finalized_model_CNN.keras")
print("Saved CNN model to finalized_model_CNN.keras")
