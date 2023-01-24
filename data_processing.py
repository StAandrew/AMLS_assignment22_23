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


def load_datasets(dataset_img_path, dataset_labels_path, filename_column_name, feature_1_column_name, feature_2_column_name, grayscale=True):
    """
    This function loads the images and labels from the dataset.
    :param dataset_img_path:        path to the folder containing the images
    :param dataset_labels_path:     path to the CSV file containing the labels
    :param label_column_name:       name of the column containing the labels
    :param filename_column_name:    name of the column containing the filenames
    :param grayscale:               if True, the images are converted to grayscale
    :return:                        labels and images
    """
    try:
        labels_df = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t").drop(['Unnamed: 0'],axis=1)
        sample_image = cv2.imread(os.path.join(dataset_img_path, labels_df.loc[0, filename_column_name]), IMREAD_COLOR)
        # sample_image = np.int8(sample_image)
        if grayscale:
            images = np.zeros((len(labels_df), sample_image.shape[0], sample_image.shape[1], 4), dtype=np.uint16)
        else:
            images = np.zeros((len(labels_df), sample_image.shape[0], sample_image.shape[1], 3, 4), dtype=np.uint16)
        i = 0
        for label_name in labels_df[filename_column_name]:
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
            i += 1
        return images, labels_df
    except:
        return None, None
    

def save_dataset(feature_arr, dataset_feature_arr_path):
    """ This function saves the labels and features to disk. 
    :param labels_df:                   dataframe containing the labels
    :param dataset_feature_arr_path:    path to the array containing the features
    """
    np.savez(dataset_feature_arr_path, feature_arr)


def load_features(dataset_features_arr_path, dataset_labels_path):
    """ This function loads the labels and features from disk.
    :param dataset_labels_path:     path to the CSV file containing the labels
    :param dataset_feature_arr_path: path to the array containing the features
    :return:                        features and labels
    """
    try:
        labels_df = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t")
        feature_arr_file = np.load(dataset_features_arr_path, allow_pickle = True)
        feature_arr = feature_arr_file['arr_0']
        feature_arr = np.array(feature_arr)
    except:
        return None, None
    return feature_arr, labels_df


def extract_face_features(images, grayscale=True):
    """
    This function extracts the face features from the images.
    :param images:      array containing the images and labels
    :return:            array containing the face features
    """
    all_features = np.zeros((len(images), 68, 2, 4))
    new_i = 0
    if grayscale:
        for i in range(len(images)):
            image = images[i, :, :, 0]
            image = img.astype(np.uint8)
            features, _ = run_dlib_shape(image)
            if features is not None:
                for j in range(len(images[i, 0, 0, :])):
                    all_features[new_i, 0, 0, j] = images[i, 0, 0, j]
                all_features[new_i, :, :, 0] = features
                new_i += 1
        all_features = all_features[:new_i]
    return all_features


def crop_resize_images_func(new_size, images, grayscale=True):
    new_w = new_size[0]
    new_h = new_size[1]

    min_x = images[0].shape[0]
    min_y = images[0].shape[1]
    max_w = 0
    max_h = 0
        
    # find the minimum and maximum width and height of the images
    for i in range(len(images)):
        if grayscale:
            image = images[i, :, :, 0]
            image = image.astype(np.uint8)
        else:
            bgr_image = images[i, :, :, :, 0]
            bgr_image = bgr_image.astype(np.uint8)
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        tmp_image = image.copy()
        tmp_image[150:300, 160:340] = 0
        tmp_image[300:370, 220:280] = 0
        _, im = cv2.threshold(tmp_image, 225, 255, cv2.THRESH_BINARY_INV)
        contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            break
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w
    # print(min_x, min_y, max_w, max_h)

    # cretate a new array for the resized images
    if grayscale:
        resized_images = np.zeros((len(images), new_h, new_w, len(images[0, 0, 0, :])), dtype=np.uint16)
    else:
        resized_images = np.zeros((len(images), new_h, new_w, 3, len(images[0, 0, 0, 0, :])), dtype=np.uint16)

    # resize images
    for i in range(len(images)):
        if grayscale:
            image = images[i, :, :, 0]
            image = image.astype(np.uint8)
            cropped_image = image[min_y:min_y+max_h, min_x:min_x+max_w]
            resized_image = cv2.resize(cropped_image, new_size)
            resized_images[i, :, :, 0] = resized_image
            for j in range(len(images[i, 0, 0, :])):
                resized_images[i, 0, 0, j] = images[i, 0, 0, j]
        else:
            image = images[i, :, :, :, 0]
            image = image.astype(np.uint8)
            cropped_image = image[min_y:min_y+max_h, min_x:min_x+max_w]
            resized_image = cv2.resize(cropped_image, new_size)
            resized_images[i, :, :, :, 0] = resized_image
            for j in range(len(images[i, 0, 0, 0, :])):
                resized_images[i, 0, 0, 0, j] = images[i, 0, 0, 0, j]
    # image = cv2.resize(image, (128, 128))
    return resized_images


def extract_eye_rectangle_remove_glasses(logger, images, eye_rect, black_rect, grayscale=True):
    if grayscale:
        logger.error("Cannot run this extract eye rectabgle function on grayscale images. Please set grayscale=False.")
        exit()

    eye_rect = (248, 275, 190, 225)
    # black_rect = (245, 280, 225, 275)
    eye_array = np.zeros((len(images), eye_rect[1]-eye_rect[0], eye_rect[3]-eye_rect[2], 3, len(images[0, 0, 0, 0, :])), dtype=np.uint16)

    new_i = 0
    for i in range(len(images)):
        image = images[i, :, :, :, 0]
        image = image.astype(np.uint8)
        eye_img = image.copy()
        eye_img = eye_img[eye_rect[0]:eye_rect[1], eye_rect[2]:eye_rect[3]]
        avg = np.mean(eye_img)
        if avg > 60:
            for j in range(len(images[0, 0, 0, 0, :])):
                eye_array[new_i, 0, 0, 0, j] = images[i, 0, 0, 0, j]
            eye_array[new_i, :, :, :, 0] = eye_img
            new_i += 1
    eye_array = eye_array[:new_i, :, :, :, :]
    return eye_array


def check_if_dataset_present(dataset_img_path, dataset_labels_path, filename_column_name):
    try:
        if os.path.exists(dataset_img_path) and os.path.exists(dataset_labels_path):
            img_dir = os.listdir(dataset_img_path)
            dataset_labels = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t").drop(['Unnamed: 0'],axis=1)
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


def extract_jawline_features(feature_arr):
    """
    This function extracts the jawline features from the landmark features.
    :param feature_arr:     array containing the landmark features
    :return:                array containing the jawline features
    """
    start_index = 0
    end_index = 16
    jawline_arr = np.zeros((len(feature_arr), end_index-start_index+1, 2, 4))
    for i in range(len(feature_arr)):
        jawline_arr[i, :, :, :] = feature_arr[i][start_index:end_index+1]
        for j in range(len(feature_arr[i, 0, 0, :])):
            jawline_arr[i, 0, 0, j] = feature_arr[i, 0, 0, j]
    return jawline_arr


def extract_smile_features(feature_arr):
    """
    This function extracts the smile features from the landmark features.
    :param feature_arr:     array containing the landmark features
    :return:                array containing the smile features
    """
    start_index = 48
    end_index = 67
    smile_arr = np.zeros((len(feature_arr), end_index-start_index+1, 2, 4))
    for i in range(len(feature_arr)):
        smile_arr[i, :, :, :] = feature_arr[i][start_index:end_index+1]
        for j in range(len(feature_arr[i, 0, 0, :])):
            smile_arr[i, 0, 0, j] = feature_arr[i, 0, 0, j]
    return smile_arr


def shuffle_split(images_dataset, labels_df, filename_column_name, feature_1_column_name, feature_2_column_name, needed_feature_column_name, test_size, logger, grayscale=True):
    """
    This function splits the dataset into a training and a test set.
    :param images_dataset:      array containing the images
    :param labels_df:           dataframe containing the labels
    :param column_name:         name of the column containing the labels
    :param test_size:           percentage of the dataset to be used as test set
    :return:                    training and test set
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
    """
    This function reshapes the data to be used in the SVM.
    :param data:    array containing the data
    :return:        reshaped array
    """
    nsamples, nx, ny = data.shape
    reshaped_data = data.reshape(nsamples,nx*ny)
    return reshaped_data


def shuffle_split_into_batches():
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


def data_split_preparation(data_dir, labels_path, filename_column, target_column, img_size, validation_size, batch_size=16, color_mode="grayscale"):
    labels_df = pd.read_csv(labels_path, sep="\t", engine="python", header=0, dtype='str')
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_size)
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
        class_mode='categorical',
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
        class_mode='categorical',
    )
    return training_batches, validation_batches


def data_preparation(data_dir, labels_path, filename_column, target_column, img_size, batch_size=16, color_mode="grayscale"):
    labels_df = pd.read_csv(labels_path, sep="\t", engine="python", header=0, dtype='str')
    image_generator = ImageDataGenerator(rescale=1. / 255.)
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
        class_mode='categorical',
    )
    return batches


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
