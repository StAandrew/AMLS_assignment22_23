from helper_utils import *
import cv2
from cv2 import IMREAD_COLOR
import dlib
import os


def load_datasets(
    dataset_img_path, dataset_labels_path, label_column_name, filename_column_name
):
    """ """
    labels_df = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t")
    labels = labels_df[label_column_name].values
    images = []
    for label_name in labels_df[filename_column_name]:
        img = cv2.imread(os.path.join(dataset_img_path, label_name), IMREAD_COLOR)
        images.append(img)
    return labels, images
