import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# -----------------------------
# 1. Dataset Loading
# -----------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)   # Standard size for pretrained models
BATCH_SIZE = 32
EPOCHS = 10

classes = ['diseased', 'mild', 'normal']

def load_dataset(dataset_dir, img_size, classes):
    X, y = [], []
    print("🔍 Loading dataset...")

    for label_index, class_name in enumerate(classes):
        img_files = glob.glob(os.path.join(dataset_dir, class_name, "*.jpg"))
        for img_path in tqdm(img_files, desc=f"Processing {class_name}"):
            try:
                # force RGB to avoid 1-channel error
                img = load_img(img_path, target_size=img_size, color_mode="rgb")
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(label_index)
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
    return np.array(X), np.array(y)

X, y = load_dataset(DATASET_DIR, IMG_SIZE, classes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train_cat = to_categorical(y_train, num_classes=len(classes))
y_test_cat = to_categorical(y_test, num_classes=len(classes))

# -----------------------------
# 2. Model Builder Functions
# -----------------------------
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_resnet(input_shape, num_classes):
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(input_shape[0], input_shape[1], 3),
                    pooling="avg")
    base.trainable = False
    model = models.Sequential([
        base,
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def build_inceptionnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    base = InceptionV3(weights="imagenet", include_top=False,
                       input_tensor=inputs, pooling="avg")
    base.trainable = False
    x = base.output
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_mobilenet(input_shape, num_classes):
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(input_shape[0], input_shape[1], 3),
                       pooling="avg")
    base.trainable = False
    model = models.Sequential([
        base,
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

# -----------------------------
# 3. Train & Evaluate
# -----------------------------
def plot_history(history, name):
    """Plot accuracy and loss curves for a model."""
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Val")
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Val")
    plt.title(f"{name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{name}_training_curves.png")
    plt.close()

def train_and_evaluate(model, name):
    print(f"\n🚀 Training {name}...")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    print(f"✅ {name} done. Accuracy: {metrics['accuracy']:.4f}")
    
    model.save(f"{name}_model.keras")
    plot_history(history, name)  # save training progress
    return metrics, history.history

results = {}

# CNN
cnn_model = build_cnn((IMG_SIZE[0], IMG_SIZE[1], 3), len(classes))
results["CNN"], _ = train_and_evaluate(cnn_model, "CNN")

# ResNet
resnet_model = build_resnet((IMG_SIZE[0], IMG_SIZE[1], 3), len(classes))
results["ResNet"], _ = train_and_evaluate(resnet_model, "ResNet")

# InceptionNet
inception_model = build_inceptionnet((IMG_SIZE[0], IMG_SIZE[1], 3), len(classes))
results["InceptionNet"], _ = train_and_evaluate(inception_model, "InceptionNet")

# MobileNet
mobile_model = build_mobilenet((IMG_SIZE[0], IMG_SIZE[1], 3), len(classes))
results["MobileNet"], _ = train_and_evaluate(mobile_model, "MobileNet")

# -----------------------------
# 4. Save Results
# -----------------------------
np.save("model_results.npy", results)
np.save("class_names.npy", classes)

print("\n✅ Training complete. Metrics saved. Training curves saved as PNGs.")
