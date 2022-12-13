import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
from cv2 import IMREAD_GRAYSCALE


# Create paths for cross-os support
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
celeba_dataset_root_path = os.path.join(
    parent_dir, "Datasets", "dataset_AMLS_22-23", "celeba"
)
celeba_dataset_img_path = os.path.join(celeba_dataset_root_path, "img")
celeba_dataset_labels_path = os.path.join(celeba_dataset_root_path, "labels.csv")

# Load the labels
labels = pd.read_csv(celeba_dataset_labels_path, skipinitialspace=True, sep="\t")

# Load the images
image_read = []
for image_name in labels["img_name"]:
    image = cv2.imread(
        os.path.join(celeba_dataset_img_path, image_name), IMREAD_GRAYSCALE
    )
    image_read.append(image)

images = np.array(image_read)
n_samples, nx, ny = images.shape
images_dataset = images.reshape(n_samples, nx * ny)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    images_dataset, labels["gender"].values, test_size=0.2, random_state=0
)
print("Loaded. Starting training...")

# Create and train SVM model
model = svm.SVC(kernel="linear")
model.fit(X_train, y_train)

# Evaluate the model on the test set
print("Training finished. Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:2f}%".format(accuracy * 100))
