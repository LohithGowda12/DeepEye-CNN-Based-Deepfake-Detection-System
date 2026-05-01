import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 128

# Load trained model
model = tf.keras.models.load_model("models/deepfake_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print("🟥 FAKE")
    else:
        print("🟩 REAL")

# Test image
predict_image("data/real/real1.jpg")  # change this path