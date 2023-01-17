import os
import tensorflow as tf

from helper_utils import *
from data_import import *

# from A1.a1 import A1
# from A2.a2 import A2
# from B1.b1 import B1
# from B2.b2 import B2

initial_config()

#  A1
labels, images = load_datasets(
    celeba_train_img_dir, celeba_train_label_dir, "gender", "img_name"
)
test_labels, test_images = load_datasets(
    celeba_test_img_dir, celeba_test_label_dir, "gender", "img_name"
)


# A2
labels, images = load_datasets(
    celeba_train_img_dir, celeba_train_label_dir, "smiling", "img_name"
)
test_labels, test_images = load_datasets(
    celeba_test_img_dir, celeba_test_label_dir, "smiling", "img_name"
)


# B1
labels, images = load_datasets(
    cartoon_train_img_dir, cartoon_train_label_dir, "face_shape", "file_name"
)
test_labels, test_images = load_datasets(
    cartoon_test_img_dir, cartoon_test_label_dir, "face_shape", "file_name"
)


# B2
labels, images = load_datasets(
    cartoon_train_img_dir, cartoon_train_label_dir, "eye_color", "file_name"
)
test_labels, test_images = load_datasets(
    cartoon_test_img_dir, cartoon_test_label_dir, "eye_color", "file_name"
)
