import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model, Model

# -----------------------------
# 5. Advanced Visualization
# -----------------------------
print("\n🎨 Generating advanced visualizations...")

# --- Load the best model (assuming MobileNet was the best) ---
best_model_name = "MobileNet"
model = load_model(f"finalized_model_{best_model_name}.keras")
classes = np.load("class_names.npy") # Load class names saved earlier

# --- Generate Predictions for Visualization ---
y_pred = np.argmax(model.predict(X_test), axis=1)


## 1. Confusion Matrix
# -----------------------------
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title(f'Confusion Matrix for {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f"{best_model_name}_confusion_matrix.png")
plt.close()
print(f"✅ Confusion matrix saved to {best_model_name}_confusion_matrix.png")


## 2. t-SNE Plot
# -----------------------------
print("\nCreating t-SNE plot... (this might take a moment)")
# Create a new model to extract features from the layer before the final classifier
feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Get the feature vectors for the test set
features = feature_extractor.predict(X_test)

# Perform t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y_test, cmap='viridis', alpha=0.7)
plt.title(f't-SNE Visualization of Features from {best_model_name}')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{best_model_name}_tsne_plot.png")
plt.close()
print(f"✅ t-SNE plot saved to {best_model_name}_tsne_plot.png")

print("\n🎉 All visualizations complete.")