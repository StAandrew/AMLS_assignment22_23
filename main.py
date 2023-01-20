from helper_utils import *
from data_processing import *

from A1.a1 import A1
# from A2.a2 import A2
# from B1.b1 import B1
# from B2.b2 import B2

logger = initial_config()

#  prepare training and verification data for A1 & A2
celeba_features, labels_df = load_datasets(celeba_features_train_dir, celeba_train_label_dir)
if celeba_features is None or labels_df is None:
    logger.info("Celeba features not found, loading raw data and extracting features...")
    images, labels_df = load_raw_datasets(celeba_train_img_dir, celeba_train_label_dir, "img_name")
    celeba_features = extract_face_features(images)
    save_dataset(celeba_features, celeba_features_train_dir)
jawline_arr = extract_jawline_features(celeba_features)
smile_arr = extract_smile_features(celeba_features)

test_size = 0.15
jawline_data_train, jawline_data_verify, jawline_label_train, jawline_label_verify = shuffle_split(jawline_arr, labels_df, "gender", test_size, logger)
smile_data_train, smile_data_verify, smile_label_train, smile_label_verify = shuffle_split(smile_arr, labels_df, "smiling", test_size, logger)

# prepare test data for A1 & A2
celeba_features_test, labels_df_test = load_datasets(celeba_features_test_dir, celeba_test_label_dir)
if celeba_features_test is None or labels_df_test is None:
    logger.info("Celeba test features not found, loading raw data and extracting features...")
    images, labels_df_test = load_raw_datasets(celeba_test_img_dir, celeba_test_label_dir, "img_name")
    celeba_features_test = extract_face_features(images)
    save_dataset(celeba_features_test, celeba_features_test_dir)
jawline_arr_test = extract_jawline_features(celeba_features_test)
smile_arr_test = extract_smile_features(celeba_features_test)

jawline_data_test, _, jawline_label_test, _ = shuffle_split(jawline_arr_test, labels_df_test, "gender", 0, logger)
smile_data_test, _, smile_label_test, _ = shuffle_split(smile_arr_test, labels_df_test, "smiling", 0, logger)

# A1
jawline_data_train = data_reshape(jawline_data_train)
jawline_data_verify = data_reshape(jawline_data_verify)
jawline_data_test = data_reshape(jawline_data_test)

a1 = A1()
a1.train(jawline_data_train, jawline_label_train)
verify_accuracy = a1.test(jawline_data_verify, jawline_label_verify)
print("A1 accuracy on verification data: {:2f}%".format(verify_accuracy * 100))
test_accuracy = a1.test(jawline_data_test, jawline_label_test)
print("A1 accuracy on test data: {:2f}%".format(test_accuracy * 100))

# A2
smile_data_train = data_reshape(smile_data_train)
smile_data_verify = data_reshape(smile_data_verify)
smile_data_test = data_reshape(smile_data_test)

print(smile_data_train.shape)
print(smile_data_verify.shape)
print(smile_data_test.shape)
print(smile_label_train.shape)
print(smile_label_verify.shape)
print(smile_label_test.shape)

a2 = A1()
a2.train(smile_data_train, smile_label_train)
verify_accuracy = a2.test(smile_data_verify, smile_label_verify)
print("A2 accuracy on verification data: {:2f}%".format(verify_accuracy * 100))
test_accuracy = a2.test(smile_data_test, smile_label_test)
print("A2 accuracy on test data: {:2f}%".format(test_accuracy * 100))


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
