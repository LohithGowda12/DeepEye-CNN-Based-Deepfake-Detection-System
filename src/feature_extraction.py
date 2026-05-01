import os
import cv2
import numpy as np

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load DenseNet201 (Pretrained)
base_model = DenseNet201(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

model = Model(inputs=base_model.input, outputs=x)

# Freeze layers (FL strategy - as in paper)
for layer in base_model.layers:
    layer.trainable = False


def load_images_and_labels(data_dir="data"):
    images = []
    labels = []

    for label, category in enumerate(["real", "fake"]):
        path = os.path.join(data_dir, category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))

            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def extract_features(images):
    images = preprocess_input(images)
    features = model.predict(images, verbose=1)
    return features