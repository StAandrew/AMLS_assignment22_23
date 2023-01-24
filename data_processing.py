""" Data processing functions.

This file contains the functions used to process the data.
Some of the functions are taken from the dlib's documentation and credit is give in function docuemntation.

Face detector can find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.
File 'shape_predictor_68_face_landmarks.dat' is necessary can be downloaded from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
The face detector is made using the classic Histogram of Oriented Gradients (HOG) feature combined with a linear classifier, an image pyramid, and sliding window detection scheme.
The pose estimator was created by using dlib's implementation of the paper:
One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and Josephine Sullivan, CVPR 2014
and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
    C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
    300 faces In-the-wild challenge: Database and results.
    Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
"""

from helper_utils import *
import cv2
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
import dlib
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    """This function converts dlib's shape to a NumPy array.

    Default function from dlib's documentation.
    Takes the (x, y)-coordinates for the facial landmarks and converts them to a NumPy array
    of (x, y)-coordinates. Returns a NumPy array of (x, y)-coordinates.

    :param shape: dlib's shape.
    :param dtype: Data type of the output array.
    :return: coords: NumPy array of (x, y)-coordinates.
    """

    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rect_to_bb(rect):
    """This function converts dlib's rectangle to a OpenCV-style bounding box.

    Default function from dlib's documentation.
    Takes a bounding predicted by dlib and converts it to the format (x, y, w, h) with OpenCV

    :param rect: dlib's rectangle.
    :return: (x, y, w, h): OpenCV-style bounding box.
    """

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def run_dlib_shape(image):
    """This function runs dlib's shape predictor on the input image.

    Default function from dlib's documentation, modified according to our needs.
    Takes a grayscale image, detects faces in it, and then loops over the face detections.
    For each face, it determines the facial landmarks for the face region, then converts the facial landmark (x, y)-coordinates to a NumPy array.
    It then converts dlib's rectangle to a OpenCV-style bounding box.
    Finally, it finds the largest face and keeps it.
    Returns the output of dlib's shape predictor and the resized image.

    :param image: Input image.
    :return: dlibout: Output of dlib's shape predictor.
    :return: resized_image: Resized image.
    """

    resized_image = image.astype("uint8")
    rects = detector(resized_image, 1)
    num_faces = len(rects)
    if num_faces == 0:
        return None, resized_image
    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)
    for (i, rect) in enumerate(rects):
        temp_shape = predictor(resized_image, rect)
        temp_shape = shape_to_np(temp_shape)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
    return dlibout, resized_image


def load_datasets(
    dataset_img_path,
    dataset_labels_path,
    filename_column_name,
    feature_1_column_name,
    feature_2_column_name,
    grayscale=True,
):
    """This function loads the images and labels from the dataset.

    The function loads the images and labels from the dataset and returns them.
    Maps -1 to 0 (for CelebA dataset) in order to use uint data type.

    Parameters
    ----------
    dataset_img_path : str
        Path to the folder containing images.
    dataset_labels_path : str
        Path to the csv file containing labels.
    filename_column_name : str
        Name of the column containing the image file names.
    feature_1_column_name : str
        Name of the column containing the first feature.
    feature_2_column_name : str
        Name of the column containing the second feature.
    grayscale : bool, optional
        Whether to load the images in grayscale or not. The default is True.
    """

    try:
        labels_df = pd.read_csv(
            dataset_labels_path, skipinitialspace=True, sep="\t"
        ).drop(["Unnamed: 0"], axis=1)
        sample_image = cv2.imread(
            os.path.join(dataset_img_path, labels_df.loc[0, filename_column_name]),
            IMREAD_COLOR,
        )
        if grayscale:
            images = np.zeros(
                (len(labels_df), sample_image.shape[0], sample_image.shape[1], 4),
                dtype=np.uint16,
            )
        else:
            images = np.zeros(
                (len(labels_df), sample_image.shape[0], sample_image.shape[1], 3, 4),
                dtype=np.uint16,
            )
        for i in range(len(labels_df)):
            label_name = labels_df.loc[i, filename_column_name]
            img = cv2.imread(os.path.join(dataset_img_path, label_name), IMREAD_COLOR)
            img = img.astype(np.uint8)
            image_number = int(labels_df.loc[i, filename_column_name][:-4])
            feature_1_label = int(labels_df.loc[i, feature_1_column_name])
            feature_2_label = int(labels_df.loc[i, feature_2_column_name])
            # LABEL MAP: -1 -> 0 in order to support uint8 format everywhere
            if feature_1_label == -1:
                feature_1_label = 0
            if feature_2_label == -1:
                feature_2_label = 0
            if grayscale:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images[i, :, :, 0] = img_gray
                images[i][0][0][1] = image_number
                images[i][0][0][2] = feature_1_label
                images[i][0][0][3] = feature_2_label
            else:
                images[i, :, :, :, 0] = img
                images[i][0][0][0][1] = image_number
                images[i][0][0][0][2] = feature_1_label
                images[i][0][0][0][3] = feature_2_label
        return images, labels_df
    except:
        return None, None


def save_dataset(feature_arr, dataset_feature_arr_path):
    """This function saves the features to disk.
    :param feature_arr:                 array containing the features
    :param dataset_feature_arr_path:    path to the array containing the features
    """
    np.savez(dataset_feature_arr_path, feature_arr)


def load_features(dataset_features_arr_path, dataset_labels_path):
    """This function loads the labels and features from disk.
    :param dataset_labels_path:         path to the CSV file containing the labels
    :param dataset_feature_arr_path:    path to the array containing the features
    :return:                            features and labels
    """
    try:
        labels_df = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t")
        feature_arr_file = np.load(dataset_features_arr_path, allow_pickle=True)
        feature_arr = feature_arr_file["arr_0"]
        feature_arr = np.array(feature_arr)
    except:
        return None, None
    return feature_arr, labels_df


def extract_face_features(images, grayscale=True):
    """This function extracts the face features from images.

    Parameters
    ----------
    images : numpy array
        Array containing images.
    grayscale : bool, optional
        Whether the images are grayscale or not. The default is True.

    Returns
    -------
    all_features : numpy array
        Array containing the face features.
    """
    all_features = np.zeros((len(images), 68, 2, 4))
    new_i = 0
    if grayscale:
        for i in range(len(images)):
            image = images[i, :, :, 0]
            image = image.astype(np.uint8)
            features, _ = run_dlib_shape(image)
            if features is not None:
                for j in range(len(images[i, 0, 0, :])):
                    all_features[new_i, 0, 0, j] = images[i, 0, 0, j]
                all_features[new_i, :, :, 0] = features
                new_i += 1
        all_features = all_features[:new_i]
    else:
        for i in range(len(images)):
            bgr_image = images[i, :, :, :, 0]
            bgr_image = bgr_image.astype(np.uint8)
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            features, _ = run_dlib_shape(image)
            if features is not None:
                for j in range(len(images[i, 0, 0, 0, :])):
                    all_features[new_i, 0, 0, j] = images[i, 0, 0, 0, j]
                all_features[new_i, :, :, 0] = features
                new_i += 1
        all_features = all_features[:new_i]
    return all_features


def extract_eye_rectangle_remove_glasses(
    logger, images, labels_df, eye_rect, grayscale=True
):
    """Extracts the eye rectangle from the images and removes the glasses from the images.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use.
    images : numpy.ndarray
        Images to extract the eye rectangle from.
    labels_df : pandas.DataFrame
        Labels dataframe.
    eye_rect : tuple
        Dimensions of eye rectangle.
    grayscale : bool, optional
        Whether the images are grayscale or not. The default is True.

    Returns
    -------
    eye_array : numpy.ndarray
        Array of eye rectangle images.
    new_labels_numbers : list
        The new labels.
    """

    if grayscale:
        logger.error(
            "Cannot run this extract eye rectangle function on grayscale images. Please set grayscale=False."
        )
        exit()
    eye_rect = (248, 275, 190, 225)
    eye_array = np.zeros(
        (
            len(images),
            eye_rect[1] - eye_rect[0],
            eye_rect[3] - eye_rect[2],
            3,
            len(images[0, 0, 0, 0, :]),
        ),
        dtype=np.uint16,
    )
    new_labels_numbers = []
    new_i = 0
    for i in range(len(images)):
        image = images[i, :, :, :, 0]
        image = image.astype(np.uint8)
        eye_img = image.copy()
        eye_img = eye_img[eye_rect[0] : eye_rect[1], eye_rect[2] : eye_rect[3]]
        avg = np.mean(eye_img)
        if avg > 60:
            for j in range(len(images[0, 0, 0, 0, :])):
                eye_array[new_i, 0, 0, 0, j] = images[i, 0, 0, 0, j]
            eye_array[new_i, :, :, :, 0] = eye_img
            new_labels_numbers.append(i)
            new_i += 1
    new_labels_df = labels_df.iloc[new_labels_numbers, :].reset_index(drop=True)
    eye_array = eye_array[:new_i, :, :, :, :]
    return eye_array, new_labels_df


def extract_jaw_rectangle_remove_beards(
    logger,
    images,
    labels_df,
    jaw_rect,
    black_rects,
    reference_colour_position,
    check_colour_position,
    remove_beards=False,
    grayscale=True,
):
    """Crops the images and removes selected areas (black_rects) from the images.

    Parameters
    ----------
    logger : logging.Logger
        Logger used for logging.
    images : numpy.ndarray
        Array of images to crop.
    labels_df : pandas.DataFrame
        Labels dataframe.
    jaw_rect : tuple
        Crop area.
    black_rects : list
        Areas to remove from the jaw images.
    reference_colour_position : tuple
        Position of the reference colour to check against.
    check_colour_position : tuple
        Position of the colour to check.
    remove_beards : bool, optional
        Whether to remove images that contain full beards or not. The default is False.
    grayscale : bool, optional
        Whether the images are grayscale or not. The default is True.

    Returns
    -------
    jaw_array : numpy.ndarray
        The jaw images.
    new_labels_numbers : list
        The new labels numbers.
    """

    if grayscale:
        jaw_array = np.zeros(
            (
                len(images),
                jaw_rect[1] - jaw_rect[0],
                jaw_rect[3] - jaw_rect[2],
                len(images[0, 0, 0, :]),
            ),
            dtype=np.uint16,
        )
        new_labels_numbers = []
        new_i = 0
        for i in range(len(images)):
            image = images[i, :, :, 0]
            image = image.astype(np.uint8)
            reference_colour = image[
                reference_colour_position[0], reference_colour_position[1]
            ]
            check_colour = image[check_colour_position[0], check_colour_position[1]]
            if not (remove_beards and np.array_equal(reference_colour, check_colour)):
                jaw_img = image.copy()
                for black_rect in black_rects:
                    jaw_img[
                        black_rect[0] : black_rect[1], black_rect[2] : black_rect[3]
                    ] = 0
                jaw_img = jaw_img[jaw_rect[0] : jaw_rect[1], jaw_rect[2] : jaw_rect[3]]
                for j in range(len(images[0, 0, 0, :])):
                    jaw_array[new_i, 0, 0, j] = images[i, 0, 0, j]
                jaw_array[new_i, :, :, 0] = jaw_img
                new_labels_numbers.append(i)
                new_i += 1
        new_labels_df = labels_df.iloc[new_labels_numbers, :].reset_index(drop=True)
        jaw_array = jaw_array[:new_i, :, :, :]
    else:
        jaw_array = np.zeros(
            (
                len(images),
                jaw_rect[1] - jaw_rect[0],
                jaw_rect[3] - jaw_rect[2],
                3,
                len(images[0, 0, 0, 0, :]),
            ),
            dtype=np.uint16,
        )
        new_labels_numbers = []
        new_i = 0
        for i in range(len(images)):
            image = images[i, :, :, :, 0]
            image = image.astype(np.uint8)
            reference_colour = image[
                reference_colour_position[0], reference_colour_position[1], :
            ]
            check_colour = image[check_colour_position[0], check_colour_position[1], :]
            if not (remove_beards and np.array_equal(reference_colour, check_colour)):
                jaw_img = image.copy()
                jaw_img[
                    black_rect1[0] : black_rect1[1], black_rect1[2] : black_rect1[3]
                ] = 0
                jaw_img[
                    black_rect2[0] : black_rect2[1], black_rect2[2] : black_rect2[3]
                ] = 0
                jaw_img = jaw_img[jaw_rect[0] : jaw_rect[1], jaw_rect[2] : jaw_rect[3]]
                for j in range(len(images[0, 0, 0, 0, :])):
                    jaw_array[new_i, 0, 0, 0, j] = images[i, 0, 0, 0, j]
                jaw_array[new_i, :, :, :, 0] = jaw_img
                new_labels_numbers.append(i)
                new_i += 1
        new_labels_df = labels_df.iloc[new_labels_numbers, :].reset_index(drop=True)
        jaw_array = jaw_array[:new_i, :, :, :, :]
    return jaw_array, new_labels_df


def check_if_dataset_present(
    dataset_img_path, dataset_labels_path, filename_column_name
):
    """Checks if the data and labels are present.

    Checks if the directories exist and are not empty.
    Checks if the labels contain the filename column.

    Parameters
    ----------
    dataset_img_path : str
        Path to the dataset images.
    dataset_labels_path : str
        Path to the dataset labels.
    filename_column_name : str
        Name of the filename column in the labels dataframe.

    Returns
    -------
    bool
        True if the dataset is present, False otherwise.
    """

    try:
        if os.path.exists(dataset_img_path) and os.path.exists(dataset_labels_path):
            img_dir = os.listdir(dataset_img_path)
            dataset_labels = pd.read_csv(
                dataset_labels_path, skipinitialspace=True, sep="\t"
            ).drop(["Unnamed: 0"], axis=1)
            if len(os.listdir(dataset_img_path)) == 0:
                return False
            if filename_column_name in dataset_labels.columns:
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        print(e)
        return False


def save_resized_images(resized_images, resized_images_path, grayscale=True):
    """Saves the resized images.

    :param resized_images:          Resized images.
    :param resized_images_path:     Path to the resized images.
    :param grayscale:               If the images are grayscale.
    """

    for i in range(len(resized_images)):
        if grayscale:
            img = resized_images[i, :, :, 0]
            img = img.astype(np.uint8)
            img_name = str(int(resized_images[i, 0, 0, 1]))
            img_path = os.path.join(resized_images_path, f"{img_name}.png")
        else:
            img = resized_images[i, :, :, :, 0]
            img = img.astype(np.uint8)
            img_name = str(int(resized_images[i, 0, 0, 0, 1]))
            img_path = os.path.join(resized_images_path, f"{img_name}.png")
        if not cv2.imwrite(img_path, img):
            raise Exception("Could not write image")
            exit(1)


def save_dataframe(logger, dataframe, dataframe_path):
    """Saves the dataframe.

    :param logger:          Logger.
    :param dataframe:       Dataframe to save.
    :param dataframe_path:  Path to the dataframe.
    """

    try:
        dataframe.to_csv(dataframe_path, sep="\t")
    except Exception as e:
        logger.error(f"Error trying to save dataframe: {e}")
        exit(1)


def extract_jawline_features(feature_arr):
    """Extracts the jawline features from the landmark features.
    :param feature_arr:     array containing the landmark features
    :return:                array containing the jawline features
    """

    start_index = 0
    end_index = 16
    jawline_arr = np.zeros((len(feature_arr), end_index - start_index + 1, 2, 4))
    for i in range(len(feature_arr)):
        jawline_arr[i, :, :, :] = feature_arr[i][start_index : end_index + 1]
        for j in range(len(feature_arr[i, 0, 0, :])):
            jawline_arr[i, 0, 0, j] = feature_arr[i, 0, 0, j]
    return jawline_arr


def extract_smile_features(feature_arr):
    """Extracts the smile features from the landmark features.
    :param feature_arr:     array containing the landmark features
    :return:                array containing the smile features
    """

    start_index = 48
    end_index = 67
    smile_arr = np.zeros((len(feature_arr), end_index - start_index + 1, 2, 4))
    for i in range(len(feature_arr)):
        smile_arr[i, :, :, :] = feature_arr[i][start_index : end_index + 1]
        for j in range(len(feature_arr[i, 0, 0, :])):
            smile_arr[i, 0, 0, j] = feature_arr[i, 0, 0, j]
    return smile_arr


def shuffle_split(
    images_dataset,
    labels_df,
    filename_column_name,
    feature_1_column_name,
    feature_2_column_name,
    needed_feature_column_name,
    test_size,
    logger,
    grayscale=True,
):
    """Splits the dataset into a training and a test set.

    Parameters
    ----------
    images_dataset : numpy.ndarray
        Dataset containing the images.
    labels_df : pandas.DataFrame
        Dataframe containing the labels.
    filename_column_name : str
        Name of the filename column in the labels dataframe.
    feature_1_column_name : str
        Name of the first feature column in the labels dataframe.
    feature_2_column_name : str
        Name of the second feature column in the labels dataframe.
    needed_feature_column_name : str
        Name of the feature column that is needed for the training.
    test_size : float
        Size of the test set.
    logger : logging.Logger
        Logger.
    grayscale : bool, optional
        Whether the images are grayscale or not. The default is True.

    Returns
    -------
    img_train : numpy.ndarray
        Training set.
    img_test : numpy.ndarray
        Test set.
    labels_train : pandas.DataFrame
        Training labels.
    labels_test : pandas.DataFrame
        Test labels.
    """

    # get the column number
    if needed_feature_column_name == filename_column_name:
        column_number = 1
    elif needed_feature_column_name == feature_1_column_name:
        column_number = 2
    elif needed_feature_column_name == feature_2_column_name:
        column_number = 3
    else:
        logger.error("Invalid column name.")
        exit()
    # calculate the number of elements in the test set
    test_element_number = int(len(images_dataset) * test_size)
    # shuffle before splitting
    np.random.shuffle(images_dataset)
    # split into training and test set
    if test_element_number != 0:
        img_train = images_dataset[:test_element_number, ...]
        img_test = images_dataset[test_element_number:, ...]
    else:
        img_train = images_dataset
        w = images_dataset[0].shape[0]
        h = images_dataset[0].shape[1]
        if grayscale:
            img_test = np.zeros((0, w, h, 1), dtype=np.uint8)
        else:
            img_test = np.zeros((0, w, h, 3, 1), dtype=np.uint8)
    # shuffle once more for good measure
    np.random.shuffle(img_train)
    np.random.shuffle(img_test)
    # add labels into another array
    label_train = np.zeros((len(img_train), 1), dtype=np.uint8)
    label_test = np.zeros((len(img_test), 1), dtype=np.uint8)
    # add the labels to the label arrays
    if grayscale:
        for i in range(len(img_train)):
            label_train[i] = img_train[i, 0, 0, column_number]
        for i in range(len(img_test)):
            label_test[i] = img_test[i, 0, 0, column_number]
        # remove the extra (label) dimension from the images array
        img_train = img_train[:, :, :, 0]
        img_test = img_test[:, :, :, 0]
    else:
        for i in range(len(img_train)):
            label_train[i] = img_train[i, 0, 0, 0, column_number]
        for i in range(len(img_test)):
            label_test[i] = img_test[i, 0, 0, 0, column_number]
        # remove the extra (label) dimension from the images array
        img_train = img_train[:, :, :, :, 0]
        img_test = img_test[:, :, :, :, 0]
    # convert the labels to a 1D array
    label_train = label_train.ravel()
    label_test = label_test.ravel()
    # cast images into uint8
    img_train = img_train.astype(np.uint8)
    img_test = img_test.astype(np.uint8)

    return img_train, img_test, label_train, label_test


def data_reshape(data):
    """This function reshapes the data to be used in the SVM.
    :param data:    array containing the data
    :return:        reshaped array
    """

    nsamples, nx, ny = data.shape
    reshaped_data = data.reshape(nsamples, nx * ny)
    return reshaped_data


def data_split_preparation(
    data_dir,
    labels_path,
    filename_column,
    target_column,
    img_size,
    validation_size,
    batch_size=16,
    color_mode="grayscale",
):
    """Prepares the data for the training and validation.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the images.
    labels_path : str
        Path to the file containing the labels.
    filename_column : str
        Name of the column containing the filenames.
    target_column : str
        Name of the column containing the labels.
    img_size : tuple
        Size of the images.
    validation_size : float
        Percentage of the dataset to be used for validation.
    batch_size : int (default: 16)
        Size of the batches.
    color_mode : str
        Color mode of the images. Can be "grayscale" or "rgb".

    Returns
    -------
    training_batches : ImageDataGenerator
        Training batches.
    validation_batches : ImageDataGenerator
        Validation batches.
    """

    labels_df = pd.read_csv(
        labels_path, sep="\t", engine="python", header=0, dtype="str"
    )
    image_generator = ImageDataGenerator(
        rescale=1.0 / 255.0, validation_split=validation_size
    )
    training_batches = image_generator.flow_from_dataframe(
        subset="training",
        dataframe=labels_df,
        directory=data_dir,
        x_col=filename_column,
        y_col=target_column,
        color_mode=color_mode,
        target_size=img_size,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="categorical",
    )
    validation_batches = image_generator.flow_from_dataframe(
        subset="validation",
        dataframe=labels_df,
        directory=data_dir,
        x_col=filename_column,
        y_col=target_column,
        color_mode=color_mode,
        target_size=img_size,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="categorical",
    )
    return training_batches, validation_batches


def data_preparation(
    data_dir,
    labels_path,
    filename_column,
    target_column,
    img_size,
    batch_size=16,
    color_mode="grayscale",
):
    """Prepares the data for the training and validation.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the images.
    labels_path : str
        Path to the file containing the labels.
    filename_column : str
        Name of the column containing the filenames.
    target_column : str
        Name of the column containing the labels.
    img_size : tuple
        Size of the images.
    batch_size : int (default: 16)
        Size of the batches.
    color_mode : str
        Whether the images are grayscale or not. The default is True. Can be "grayscale" or "rgb".
    """

    labels_df = pd.read_csv(
        labels_path, sep="\t", engine="python", header=0, dtype="str"
    )
    image_generator = ImageDataGenerator(rescale=1.0 / 255.0)
    batches = image_generator.flow_from_dataframe(
        dataframe=labels_df,
        directory=data_dir,
        x_col=filename_column,
        y_col=target_column,
        color_mode=color_mode,
        target_size=img_size,
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
    )
    return batches
