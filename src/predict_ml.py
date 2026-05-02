import cv2
import numpy as np
import joblib

from feature_extraction import extract_features

# Load trained model
model = joblib.load("model_rf.pkl")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    img = np.array([img])  # batch of 1

    features = extract_features(img)

    prediction = model.predict(features)[0]

    if prediction == 0:
        print("🟢 REAL")
    else:
        print("🔴 FAKE")


if __name__ == "__main__":
    # change this path to your test image
    predict_image("test.jpg")