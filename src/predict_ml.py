import numpy as np
import cv2
import tensorflow as tf
import joblib

IMG_SIZE = 224

# Load ML model
rf = joblib.load("models/rf_model.pkl")

# Load CNNs (same as feature extraction)
resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
xception = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
densenet = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')


def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found:", image_path)
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)

    f1 = resnet.predict(img)
    f2 = xception.predict(img)
    f3 = densenet.predict(img)

    combined = np.concatenate([f1.flatten(), f2.flatten(), f3.flatten()])

    return combined.reshape(1, -1)


def predict(image_path):
    features = extract_features(image_path)

    if features is None:
        return

    pred = rf.predict(features)[0]

    if pred == 1:
        print("🟥 FAKE")
    else:
        print("🟩 REAL")


# 🔥 Test
predict("data/real/129_4.png")  # change path