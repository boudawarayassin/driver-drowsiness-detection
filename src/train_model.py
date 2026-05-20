"""
Train CNN model for eye open/closed classification.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import *

import tensorflow as tf
import matplotlib.pyplot as plt

# ── Load datasets ──────────────────────────────────
split = VALIDATION_SPLIT + TEST_SPLIT   # 30% out

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=split,
    subset='training',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode=COLOR_MODE
)

val_full = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=split,
    subset='validation',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode=COLOR_MODE
)

# Split val → val + test
n        = tf.data.experimental.cardinality(val_full).numpy()
half     = n // 2
val_ds   = val_full.skip(half)
test_ds  = val_full.take(half)

print(f"Train: {tf.data.experimental.cardinality(train_ds)} batches")
print(f"Val:   {tf.data.experimental.cardinality(val_ds)} batches")
print(f"Test:  {tf.data.experimental.cardinality(test_ds)} batches")
print(f"Classes: {train_ds.class_names}")

# ── Normalize ──────────────────────────────────────
norm = tf.keras.layers.Rescaling(1./255)

def normalize(x, y): return norm(x), y

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(normalize).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.map(normalize).cache().prefetch(AUTOTUNE)
test_ds  = test_ds.map(normalize).cache().prefetch(AUTOTUNE)

# ── Model ──────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Callbacks ──────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True,
        monitor='val_accuracy'
    )
]

# ── Train ──────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Test evaluation ────────────────────────────────
print("\nEvaluating on test set...")
loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

# ── Graphs ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss'); ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'training_history.png'))
plt.show()
print("Done! Model saved to:", MODEL_PATH)
