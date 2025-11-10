# train_model.py
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers , models, optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
#shortcut for run a current opened file is ctrl+shift+F10
#here the accuracy is about 0.6 . since it is a low accuracy but we have lower dependency devices so that we are using only 100 images on both 2 classes image training for the model , then the model trained is saveed in the models folder
# ========== Configuration ==========
BASE_DIR = Path('data/waste_dataset')
TRAIN_DIR = BASE_DIR / 'TRAIN'
TEST_DIR = BASE_DIR / 'TEST'
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)   # Smaller image size = faster training
BATCH_SIZE = 16         # Moderate batch size
EPOCHS = 15             # Balanced number of epochs
LEARNING_RATE = 1e-4
SEED = 42

print("✅ TensorFlow version:", tf.__version__)
print("✅ Training directory:", TRAIN_DIR.resolve())
print("✅ Model save path:", MODEL_DIR.resolve())

# ========== Data Generators ==========
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class indices for app.py
classes = train_gen.class_indices
classes_json = {k: int(v) for k, v in classes.items()}
with open(MODEL_DIR / 'classes.json', 'w') as f:
    json.dump(classes_json, f)
print("✅ Saved classes.json:", classes_json)

# ========== Model Definition ==========
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # freeze base initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(classes), activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========== Callbacks ==========
ckpt_path = MODEL_DIR / 'waste_classifier_best.h5'
callbacks = [
    ModelCheckpoint(str(ckpt_path), monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

# ========== Stage 1: Train Classifier Head ==========
print("\n🧠 Stage 1: Training classifier head...")
history_1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS // 2,
    callbacks=callbacks,
    verbose=1
)

# ========== Stage 2: Fine-tuning ==========
print("\n🔧 Stage 2: Fine-tuning deeper layers...")
base_model.trainable = True

# Only unfreeze last 30% of base_model layers for moderate tuning
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_path = MODEL_DIR / 'waste_classifier.h5'
model.save(final_path)
print(f"✅ Final model saved to {final_path}")

# ========== Evaluation ==========
loss, acc = model.evaluate(test_gen)
print(f"\n📊 Test accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}")

# Save training history
hist_data = {
    'acc': history_2.history.get('accuracy', []),
    'val_acc': history_2.history.get('val_accuracy', []),
    'loss': history_2.history.get('loss', []),
    'val_loss': history_2.history.get('val_loss', [])
}
with open(MODEL_DIR / 'train_history.json', 'w') as f:
    json.dump(hist_data, f, indent=2)
print("✅ Training history saved to models/train_history.json")
