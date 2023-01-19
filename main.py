from helper_utils import *
from data_processing import *

# from A1.a1 import A1
# from A2.a2 import A2
# from B1.b1 import B1
# from B2.b2 import B2

initial_config()

#  prepare data for A1 & A2
face_features, face_labels = load_datasets(celeba_features_train_img_dir, celeba_features_train_label_dir)
if face_features is None or face_labels is None:
    images, labels_df = load_raw_datasets(celeba_train_img_dir, celeba_train_label_dir, "img_name")
    face_features, face_labels = extract_face_features(images, labels_df)
    save_datasets(face_features, face_labels, celeba_features_train_label_dir, celeba_features_train_img_dir)
jawline_arr = extract_jawline_features(face_features)
smile_arr = extract_smile_features(face_features)

batch_size = 30
jawline_train_batches, jawline_verify_batches, jawline_test_batches = shuffle_split_into_batches(jawline_arr, face_labels, "gender", batch_size)


exit()
test_labels, test_images = load_raw_datasets(celeba_test_img_dir, celeba_test_label_dir, "gender", "img_name")


# A2
labels, images = load_raw_datasets(celeba_train_img_dir, celeba_train_label_dir, "smiling", "img_name")
test_labels, test_images = load_raw_datasets(celeba_test_img_dir, celeba_test_label_dir, "smiling", "img_name")


# B1
labels, images = load_daload_raw_datasetstasets(cartoon_train_img_dir, cartoon_train_label_dir, "face_shape", "file_name")
test_labels, test_images = load_raw_datasets(cartoon_test_img_dir, cartoon_test_label_dir, "face_shape", "file_name")


# B2
labels, images = load_raw_datasets(cartoon_train_img_dir, cartoon_train_label_dir, "eye_color", "file_name", grayscale=False)
test_labels, test_images = load_raw_datasets(cartoon_test_img_dir, cartoon_test_label_dir, "eye_color", "file_name", grayscale=False)
