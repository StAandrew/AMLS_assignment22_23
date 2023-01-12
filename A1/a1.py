import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
from cv2 import IMREAD_GRAYSCALE, IMREAD_COLOR
import time

VERIFICATION_SPLIT = 0.2


def load_data(root_path):
    # Create paths
    dataset_img_path = os.path.join(root_path, "img")
    dataset_labels_path = os.path.join(root_path, "labels.csv")
    # Load the labels
    labels = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t")
    # Load the images
    image_read = []
    for image_name in labels["img_name"]:
        image = cv2.imread(os.path.join(dataset_img_path, image_name), IMREAD_COLOR)
        image_read.append(image)

    images = np.array(image_read)
    return labels, images


# Create paths for cross-os support
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load development and test datasets
dataset_root_path = os.path.join(parent_dir, "Datasets", "celeba")
labels, images = load_data(dataset_root_path)

test_root_path = os.path.join(parent_dir, "Datasets", "celeba_test")
test_labels, test_images = load_data(test_root_path)

n_samples, img_height, img_width, n_channels = images.shape
images_dataset = images.reshape(n_samples, img_height * img_width * n_channels)

# Split the data into training and test sets
img_train, img_verify, label_train, label_verify = train_test_split(
    images_dataset,
    labels["gender"].values,
    test_size=VERIFICATION_SPLIT,
    random_state=0,
    shuffle=True,
)
img_train = np.array(img_train)
img_verify = np.array(img_verify)
label_train = np.array(label_train)
label_verify = np.array(label_verify)

print("Loaded. Starting training...")
# exit(0)
time.sleep(3)

# Create and train SVM model
model = svm.SVC(kernel="linear")
model.fit(img_train, label_train)

# Evaluate the model on the test set
print("Training finished. Evaluating...")
label_pred = model.predict(img_verify)
accuracy = accuracy_score(label_verify, label_pred)
print("Accuracy: {:2f}%".format(accuracy * 100))

