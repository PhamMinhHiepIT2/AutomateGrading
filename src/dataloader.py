from pathlib import Path
import cv2
from tensorflow.keras.utils import to_categorical
import numpy as np


def load_data(data_path: str):
    dataset_dir = data_path
    images = []
    labels = []

    for img_path in Path(dataset_dir + 'unchoice/').glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
        img = img.reshape((28, 28, 1))
        label = to_categorical(0, num_classes=2)
        images.append(img / 255.0)
        labels.append(label)

    for img_path in Path(dataset_dir + 'choice/').glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
        img = img.reshape((28, 28, 1))
        label = to_categorical(1, num_classes=2)
        images.append(img / 255.0)
        labels.append(label)

    datasets = list(zip(images, labels))
    np.random.shuffle(datasets)
    images, labels = zip(*datasets)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels
