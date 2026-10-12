import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import pathlib
import json
import os
from sklearn.model_selection import train_test_split

# ==============================
# SETTINGS
# ==============================
DATASET_DIR = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

print("🔍 Loading dataset safely...")

# ==============================
# LOAD DATA SAFELY (NO GLOB ERROR)
# ==============================
data_dir = pathlib.Path(DATASET_DIR)

class_names = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
num_classes = len(class_names)

print("Classes:", class_names)

image_paths = []

# SAFE FILE SCAN (NO GLOB)
for class_folder in data_dir.iterdir():
    if class_folder.is_dir():
        for file in class_folder.iterdir():
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_paths.append(file)

images = []
labels = []
skipped = 0

for path in image_paths:
    try:
        img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img = tf.keras.utils.img_to_array(img)

        label = class_names.index(path.parent.name)

        images.append(img)
        labels.append(label)

    except:
        print("❌ Skipping bad file:", path)
        skipped += 1

print(f"Total images: {len(images)} | Skipped: {skipped}")

images = np.array(images) / 255.0
labels = np.array(labels)

# ==============================
# SPLIT DATA
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

# ==============================
# CLASS WEIGHTS (FIX IMBALANCE)
# ==============================
counts = np.bincount(y_train)
total = np.sum(counts)

class_weight = {
    i: total / (num_classes * counts[i]) for i in range(num_classes)
}

print("Class counts:", counts)

# ==============================
# MODEL (STRONG - MobileNetV2)
# ==============================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# TRAIN
# ==============================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight
)

# ==============================
# SAVE MODEL + CLASSES
# ==============================
os.makedirs("model", exist_ok=True)

model.save("model/model.h5")

with open("model/classes.json", "w") as f:
    json.dump(class_names, f)

print("\n🎉 Model saved successfully!")