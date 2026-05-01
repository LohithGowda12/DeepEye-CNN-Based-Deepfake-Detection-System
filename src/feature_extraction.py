import tensorflow as tf
import numpy as np
import cv2
import os

IMG_SIZE = 224
DATA_DIR = "data"

# Load pretrained ResNet
resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
xception = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
densenet = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')


def extract_features():
    features = []
    labels = []

    for category in ["real", "fake"]:
        path = os.path.join(DATA_DIR, category)
        label = 0 if category == "real" else 1

        for img in os.listdir(path)[:100]:
            print("Processing:", img)
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                
                img_array = np.expand_dims(img_array, axis=0)

                f1 = resnet.predict(img_array)
                f2 = xception.predict(img_array)
                f3 = densenet.predict(img_array)

                combined = np.concatenate([f1.flatten(), f2.flatten(), f3.flatten()])

                features.append(combined)
                labels.append(label)

            except:
                pass

    return np.array(features), np.array(labels)


X, y = extract_features()

print("Feature shape:", X.shape)

np.save("models/features.npy", X)
np.save("models/labels.npy", y)

print("✅ Features saved successfully!")