import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# -----------------------------
# 1. Dataset Loading (Essential Setup)
# -----------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
classes = ['diseased', 'mild', 'normal'] # Make sure this matches your subdirectories

def load_dataset(dataset_dir, img_size, classes):
    X, y = [], []
    print("🔍 Loading dataset for visualizations...")

    for label_index, class_name in enumerate(classes):
        img_files = glob.glob(os.path.join(dataset_dir, class_name, "*.jpg"))
        for img_path in tqdm(img_files, desc=f"Processing {class_name}"):
            try:
                img = load_img(img_path, target_size=img_size, color_mode="rgb")
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(label_index)
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
    return np.array(X), np.array(y)

# Load the data
X, y = load_dataset(DATASET_DIR, IMG_SIZE, classes)

# Perform the same train-test split to get the correct X_test and y_test
# Using the same random_state ensures you get the exact same test set as during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Dataset loaded. Test set has {len(X_test)} images.")

# -----------------------------
# 2. Advanced Visualization for All Models
# -----------------------------
print("\n🎨 Generating advanced visualizations for all trained models...")

model_names = ["CNN", "ResNet", "InceptionNet", "MobileNet"] 

for model_name in model_names:
    model_path = f"models/{model_name}_model.keras"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model file not found at {model_path}. Skipping.")
        continue

    print(f"\nProcessing visualizations for: {model_name}")
    
    model = load_model(model_path)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    ## Confusion Matrix
    print(f"Creating confusion matrix for {model_name}...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_filename = f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"✅ Confusion matrix saved to {cm_filename}")

    ## t-SNE Plot
    print(f"Creating t-SNE plot for {model_name}... (this might take a moment)")
    
    if model_name == "CNN":
        feature_layer_output = model.layers[-4].output
    else:
        feature_layer_output = model.layers[-2].output

    feature_extractor = Model(inputs=model.inputs, outputs=feature_layer_output)
    features = feature_extractor.predict(X_test)
    
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y_test, cmap='viridis', alpha=0.7)
    plt.title(f't-SNE Visualization of Features from {model_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.grid(True)
    plt.tight_layout()
    tsne_filename = f"{model_name}_tsne_plot.png"
    plt.savefig(tsne_filename)
    plt.close()
    print(f"✅ t-SNE plot saved to {tsne_filename}")

print("\n🎉 All visualizations complete.")