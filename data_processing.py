from helper_utils import *
import cv2
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
import dlib
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    os.path.join(root_dir, "shape_predictor_68_face_landmarks.dat")
)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype("uint8")

    # detect faces in the grayscale image
    rects = detector(resized_image, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(resized_image, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def load_raw_datasets(dataset_img_path, dataset_labels_path, filename_column_name, grayscale=True):
    """
    This function loads the images and labels from the dataset.
    :param dataset_img_path:        path to the folder containing the images
    :param dataset_labels_path:     path to the CSV file containing the labels
    :param label_column_name:       name of the column containing the labels
    :param filename_column_name:    name of the column containing the filenames
    :param grayscale:               if True, the images are converted to grayscale
    :return:                        labels and images
    """
    labels_df = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t").drop(['Unnamed: 0'],axis=1)
    # labels = labels_df[label_column_name].values
    images = []
    for label_name in labels_df[filename_column_name]:
        img = cv2.imread(os.path.join(dataset_img_path, label_name), IMREAD_COLOR)
        if grayscale:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img_gray)
        else:
            images.append(img)
    return images, labels_df
    

def save_datasets(feature_arr, labels_df, dataset_labels_path, dataset_feature_arr_path):
    """ This function saves the labels and features to disk. 
    :param labels_df:               dataframe containing the labels
    :param feature_arr:            array containing the features
    :param dataset_labels_path:     path to the CSV file containing the labels
    :param dataset_feature_arr_path: path to the array containing the features
    """
    labels_df.to_csv(dataset_labels_path, sep="\t", index=False)
    np.savez(dataset_feature_arr_path, feature_arr)


def load_datasets(dataset_features_arr_path, dataset_labels_path):
    """ This function loads the labels and features from disk.
    :param dataset_labels_path:     path to the CSV file containing the labels
    :param dataset_feature_arr_path: path to the array containing the features
    :return:                        labels and features
    """
    try:
        labels_df = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t")
        feature_arr_file = np.load(dataset_features_arr_path, allow_pickle = True)
        feature_arr = feature_arr_file['arr_0']
    except:
        return None, None
    return feature_arr, labels_df


def extract_face_features(images, labels):
    """
        Extracts face features from images where at least one face is detected.
        The images extracted are placed in a new dedicated folder and are all in 'grayscale'.
    """
    all_features = []
    all_labels = pd.DataFrame()
    for i in range(len(images)):
        features, _ = run_dlib_shape(images[i])
        if features is not None:
            all_features.append(features)
            all_labels = pd.concat([all_labels, labels.iloc[[i], :]], axis=0, ignore_index=True)
    all_features = np.array(all_features)
    return all_features, all_labels

def extract_jawline_features(feature_arr):
    """
    This function extracts the jawline features from the landmark features.
    :param feature_arr:     array containing the landmark features
    :return:                array containing the jawline features
    """
    start_index = 0
    end_index = 16
    jawline_arr = []
    for i in range(len(feature_arr)):
        jawline_arr.append(feature_arr[i][start_index:end_index+1])
    jawline_arr = np.array(jawline_arr)
    return jawline_arr


def extract_smile_features(feature_arr):
    """
    This function extracts the smile features from the landmark features.
    :param feature_arr:     array containing the landmark features
    :return:                array containing the smile features
    """
    start_index = 48
    end_index = 64
    smile_arr = []
    for i in range(len(feature_arr)):
        smile_arr.append(feature_arr[i][start_index:end_index+1])
    return smile_arr


def shuffle_split_into_batches(images_dataset, labels_dataset, column_name, batch_size):
    """
    This function splits the dataset into batches using the train_test_split, 
    from_tensor_slices, shuffle and batch functions.
    :param images_dataset:      array containing the images
    :param labels_dataset:      dataframe containing the labels
    :param column_name:         name of the column containing the labels
    :param batch_size:          size of the batches
    :return:                    batches of images and labels
    """
    images_dataset = np.expand_dims(images_dataset, axis=3)
    for i in range(len(images_dataset)):
        images_dataset[i][0][0][0] = int(labels_dataset.loc[i, 'img_name'][:-4])
    # Put 60% of dataset into training set
    img_train, img_rest_of_dataset, label_train, labels_rest_of_dataset = train_test_split(
        images_dataset,
        labels_dataset,
        test_size=0.40,
        random_state=42,
        shuffle=False,
    )
    print(label_train.loc[:, 'img_name'])
    exit(0)
    print(label_train.loc[100, 'img_name'][:-4])
    exit(0)
    check_ints = []
    for i in range(20):
        intx = random.randint(0, 5000)
        check_ints.append(intx)
    for i in check_ints:
        try:
            val = int(label_train.loc[i, 'img_name'][:-4])
            print(f"img_train {img_train[i][0][0][0]}")
            print(f"label {val}")
        except KeyError:
            continue
    # check_ints2 = [4000, 4264, 4628, 4756, 3758, 3768, 4045, 4134, 4516]
    # for i in check_ints2:
    #     print(img_rest_of_dataset[i][0][0][0])
    #     print(labels_rest_of_dataset.loc[i, 'img_name'][:-4])
    exit()
    # labels_rest_of_dataset.reset_index(inplace=True)
    print(img_rest_of_dataset)
    print(labels_rest_of_dataset["img_name"].values)
    print(labels_rest_of_dataset[column_name].values)
    # Put the rest of dataset into validation and test set
    img_validate, img_test, label_validate, label_test = train_test_split(
        img_rest_of_dataset,
        labels_rest_of_dataset,
        test_size=0.50,
        random_state=0,
        shuffle=False,
    )
    # print(labels_rest_of_dataset["img_name"].values)
    # print(labels_rest_of_dataset[column_name].values)
    exit()
    # make batches
    training_data = tf.data.Dataset.from_tensor_slices((img_train, label_train))
    training_batches = (
        training_data.shuffle(len(training_data)).batch(batch_size, drop_remainder=True).prefetch(1)
    )
    validation_data = tf.data.Dataset.from_tensor_slices((img_validate, label_validate))
    validation_batches = (
        validation_data.shuffle(len(validation_data)).batch(batch_size, drop_remainder=True).prefetch(1)
    )
    test_data = tf.data.Dataset.from_tensor_slices((img_test, label_test))
    test_batches = (
        test_data.shuffle(len(test_data)).batch(batch_size, drop_remainder=True).prefetch(1)
    )
    return training_batches, validation_batches, test_batches


def batches_to_arrays(batches):
    """
    This function converts the batches into arrays.
    :param batches:     batches of images and labels
    :return:            arrays of images and labels
    """
    batches = batches.unbatch()
    images = np.array(list(x for x, y in batches))
    labels = np.array(list(y for x, y in batches))
    return images, labels


# Feature extraction for training data
def extract_features_labels_train():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [
        os.path.join(img_train_dir_b, l) for l in os.listdir(img_train_dir_b)
    ]
    target_size = None
    # Convert CSV into dataframe (tab seperators and header column defined)
    labels_file = pd.read_csv(
        os.path.join(basedir, labels_train_dir), sep="\t", engine="python", header=0
    )
    # Extract gender labels from df
    smiling_labels_df = labels_file["smiling"]
    smiling_labels = smiling_labels_df.values
    if os.path.isdir(img_train_dir_b):
        all_features = []
        all_labels = []

        for img_path in image_paths:
            file_name = img_path.split(".")[-2].split("/")[-1]
            # load image
            img = image.img_to_array(
                image.load_img(
                    img_path, target_size=target_size, interpolation="bicubic"
                )
            )
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(smiling_labels[int(file_name)])

    landmark_features = np.array(all_features)
    smiling_labels = (
        np.array(all_labels) + 1
    ) / 2  # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, smiling_labels


# Feature extracion for testing data
def extract_features_labels_test():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(img_test_dir_b, l) for l in os.listdir(img_test_dir_b)]
    target_size = None
    # Convert CSV into dataframe (tab seperators and header column defined)
    labels_file = pd.read_csv(labels_test_dir, sep="\t", engine="python", header=0)
    # Extract gender labels from df
    smiling_labels_df = labels_file["smiling"]
    smiling_labels = smiling_labels_df.values
    if os.path.isdir(img_test_dir):
        all_features = []
        all_labels = []

        for img_path in image_paths:
            file_name = img_path.split(".")[-2].split("/")[-1]
            # load image
            img = image.img_to_array(
                image.load_img(
                    img_path, target_size=target_size, interpolation="bicubic"
                )
            )
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(smiling_labels[int(file_name)])

    landmark_features = np.array(all_features)
    smiling_labels = (
        np.array(all_labels) + 1
    ) / 2  # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, smiling_labels