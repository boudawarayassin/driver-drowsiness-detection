"""
Model evaluation — confusion matrix, classification report, wrong predictions.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow as tf

# ── Load model ─────────────────────────────────────
model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# ── Load test dataset ──────────────────────────────
split   = VALIDATION_SPLIT + TEST_SPLIT
val_full = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=split,
    subset='validation',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode=COLOR_MODE
)

n       = tf.data.experimental.cardinality(val_full).numpy()
test_ds = val_full.take(n // 2)

norm    = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (norm(x), y))

class_names = val_full.class_names
print("Classes:", class_names)

# ── Collect predictions ────────────────────────────
y_true, y_pred = [], []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    p     = (preds > 0.5).astype(int)
    y_true.extend(labels.numpy())
    y_pred.extend(p.flatten())

# ── Results ────────────────────────────────────────
print(f"\nTotal test images: {len(y_true)}")

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Confusion matrix plot ──────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — Test Set')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'confusion_matrix.png'))
plt.show()

# ── Wrong predictions ──────────────────────────────
print("\nShowing wrong predictions...")
for images, labels in test_ds.take(1):
    preds = model.predict(images, verbose=0)
    p     = (preds > 0.5).astype(int)
    for i in range(len(images)):
        if p[i] != labels[i]:
            plt.figure(figsize=(3, 3))
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(f"True: {class_names[labels[i]]}  "
                      f"Pred: {class_names[p[i][0]]}")
            plt.axis('off')
            plt.show()
