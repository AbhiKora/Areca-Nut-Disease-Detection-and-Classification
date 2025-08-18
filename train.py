import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.optimizers import Adam

# =========================================================
# CONFIG
# =========================================================
DATASET_DIR = "X-ArecaNet"   # path to unzipped dataset
BATCH_SIZE = 32
IMG_SIZE = (128, 128)  # For CNN
IMG_SIZE_BIG = (224, 224)  # For pretrained models
EPOCHS = 10

# =========================================================
# DATA LOADING
# =========================================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2
)

datagen_rgb = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=lambda x: np.repeat(x, 3, axis=2) if x.shape[2] == 1 else x
)

# CNN → grayscale (1 channel)
train_gen_gray = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen_gray = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Pretrained models → RGB (3 channels, 224x224)
train_gen_rgb = datagen_rgb.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE_BIG,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen_rgb = datagen_rgb.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE_BIG,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

num_classes = len(train_gen_gray.class_indices)

# =========================================================
# MODEL DEFINITIONS
# =========================================================

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    return model

def build_mobilenetv2(num_classes):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
    base.trainable = False  # Freeze base
    model = Sequential([
        base,
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model

def build_efficientnet(num_classes):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
    base.trainable = False
    model = Sequential([
        base,
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="sigmoid")
    ])
    return model

def build_resnet50(num_classes):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
    base.trainable = False
    model = Sequential([
        base,
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model

# =========================================================
# TRAINING + EVALUATION
# =========================================================
def train_and_evaluate(model, train_gen, val_gen, name, epochs=EPOCHS):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)

    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen)
    y_pred = model.predict(val_gen)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    y_true = val_gen.classes

    acc = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average="weighted")
    recall = recall_score(y_true, y_pred_classes, average="weighted")
    f1 = f1_score(y_true, y_pred_classes, average="weighted")

    print(f"\n{name} Results:")
    print(classification_report(y_true, y_pred_classes, target_names=list(train_gen.class_indices.keys())))

    # Save model
    model.save(f"{name}.keras")

    return {
        "name": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =========================================================
# RUN ALL MODELS
# =========================================================
results = []

# Custom CNN
# cnn_model = build_cnn((128,128,1), num_classes)   # <-- fixed input
# results.append(train_and_evaluate(cnn_model, train_gen_gray, val_gen_gray, "CNN"))

# # MobileNetV2
# mobilenet_model = build_mobilenetv2(num_classes)
# results.append(train_and_evaluate(mobilenet_model, train_gen_rgb, val_gen_rgb, "MobileNetV2"))

# EfficientNetB0
efficientnet_model = build_efficientnet(num_classes)
results.append(train_and_evaluate(efficientnet_model, train_gen_rgb, val_gen_rgb, "EfficientNetB0"))

# ResNet50
resnet_model = build_resnet50(num_classes)
results.append(train_and_evaluate(resnet_model, train_gen_rgb, val_gen_rgb, "ResNet50"))

# =========================================================
# PLOT RESULTS
# =========================================================
names = [r["name"] for r in results]
accs = [r["accuracy"] for r in results]
f1s = [r["f1"] for r in results]

plt.figure(figsize=(8,5))
plt.bar(names, accs, color="skyblue", label="Accuracy")
plt.bar(names, f1s, alpha=0.6, label="F1-score")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.show()
