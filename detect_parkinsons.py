from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
# We import features to use HOG(Histogram of Oriented Gradients) for the image.
from skimage import feature
from imutils import build_montages
from imutils import paths

import os
import numpy as np
import argparse
import cv2

dataset_path_spiral = "./dataset/spiral"
dataset_path_wave = "./dataset/wave"


def process_image(image):
    return feature.hog(image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")


def create_dataset(path):
    imagePaths = list(paths.list_images(path))
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    # print(imagepath)

    for impath in imagePaths:
        print(impath)
        datasplit = impath.split("/")[-3]
        label = impath.split("/")[-2]
        image = cv2.imread(impath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features = process_image(image)
        if datasplit == "training":
            training_data.append(features)
            training_labels.append(label)
        else:
            testing_data.append(features)
            testing_labels.append(label)

    return np.array(training_data), np.array(training_labels), np.array(testing_data), np.array(testing_labels)


def train(trainX, trainy):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(trainX, trainy)
    return model


def test_metrics(model, testX, testy):
    predictions = model.predict(testX)
    metrics = {}

    cm = confusion_matrix(testy, predictions).flatten()
    (tn, fp, fn, tp) = cm
    metrics["acc"] = (tp + tn)/(float(cm.sum()))
    metrics["sensitivity"] = tp / float(tp + fn)
    metrics["specificity"] = tn / float(tn + fp)

    return metrics


if __name__ == "__main__":
    trainX, trainy, testX, testy = create_dataset(dataset_path_spiral)
    print(len(trainX), len(trainy), len(testX), len(testy))
    model = train(trainX, trainy)
    metrics = test_metrics(model, testX, testy)
    print(metrics)
