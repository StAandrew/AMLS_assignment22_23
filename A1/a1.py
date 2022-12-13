import numpy as np
import pandas as pd
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
labels = pd.read_csv(celeba_dataset_labels_path, skipinitialspace=True, sep='\t')

# Load the images
images = []
for image_name in labels['img_name']:
    image = cv2.imread(os.path.join(celeba_dataset_img_path, image_name), IMREAD_GRAYSCALE)
    images.append(image)
