import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

print("🚀 DeepEye Started")

DATA_DIR = "data"
IMG_SIZE = 128

data = []
labels = []

# Load images
for category in ["real", "fake"]:
    path = os.path.join(DATA_DIR, category)
    label = 0 if category == "real" else 1

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            
            data.append(img_array)
            labels.append(label)
        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

print("Data loaded:", len(data))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=5)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Save model
model.save("models/deepfake_model.h5")

print("✅ Model trained and saved!")