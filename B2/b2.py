import tensorflow as tf
import numpy as np
import os
import dlib
import cv2
from cv2 import IMREAD_COLOR
from keras.utils import image_utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import optimizers
from helper_utils import plot_performance, plot_confusion_matrix, rect_to_bb, root_dir

import time


# MNIST dataset parameters.
num_classes = 5  # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.001
training_steps = 78
batch_size = 128
display_step = 10
epochs = 10
num_samples = 100  # -1 for all available samples
train_model = False
save_model = True  # only save if train_model is True
validation_split = 0.2

# Network parameters.
conv1_filters = 32  # number of filters for 1st conv layer.
conv2_filters = 64  # number of filters for 2nd conv layer.
fc1_units = 1024  # number of neurons for 1st fully-connected layer.


def load_data(root_path):
    # Create paths
    dataset_img_path = os.path.join(root_path, "img")
    dataset_labels_path = os.path.join(root_path, "labels.csv")
    # Load the labels
    labels_df = pd.read_csv(dataset_labels_path, skipinitialspace=True, sep="\t")
    # Adjust size if needed
    if num_samples > 0:
        labels_df = labels_df[0:num_samples]
    else:
        labels_df = labels_df
    labels = labels_df["eye_color"].values
    # Load the images
    image_read = []
    all_features = []
    for image_name in labels_df["file_name"]:
        img = cv2.imread(os.path.join(dataset_img_path, image_name), IMREAD_COLOR)
        img_features, _ = run_dlib_shape(img)
        if features is not None:
            all_features.append(img_features)

            all_labels.append(face_labels[file_name])
            print(file_name)
        resized_image = cv2.resize(
            frame, model_image_size, interpolation=cv2.INTER_CUBIC
        )
        cv2.imwrite(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "step_zero", f"{0}.png"
            ),
            frame,
        )
        cv2.imwrite(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "step_one", f"{0}.png"
            ),
            resized_image,
        )

        resized_image = resized_image.astype(np.float32)

        resized_image /= 255.0
        cv2.imwrite(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "step_two", f"{0}.png"
            ),
            resized_image * 255.0,
        )

        exit(0)
        images = np.array(image_read, np.float32)

        # image = image_utils.img_to_array(
        #     image_utils.load_img(
        #         os.path.join(dataset_img_path, image_name),
        #         target_size=(500, 500),
        #         # interpolation="bicubic",
        #     )
        # )
        # image_resized = image.astype("uint8")
        image_read.append(image)

    landmark_features = np.array(all_features)
    face_labels = np.array(
        all_labels
    )  # simply converts the -1 into 0, so male=0 and female=1

    # Convert to float and normalize images value from [0, 255] to [0, 1].
    images = np.array(image_read, np.float32)

    labels = np.array(labels)

    # image_data_generator = ImageDataGenerator(
    #     rescale=1.0/255.0, validation_split=validation_split, horizontal_flip=True
    # )
    # training_batches = image_generator.flow_from_dataframe(
    #     dataframe=training_labels,
    #     directory=training_dir,
    #     x_col=filename_column,
    #     y_col=target_column,
    #     subset="training",
    #     batch_size=batches_size,
    #     seed=42,
    #     color_mode=color_mode,
    #     shuffle=True,
    #     target_size=img_size,
    # )

    return images, labels


def get_training_data(root_path):
    dataset_root_path = os.path.join(root_path, "Datasets", "cartoon_set")
    images, labels = load_data(dataset_root_path)
    return images, labels


def get_testing_data(root_path):
    test_root_path = os.path.join(root_path, "Datasets", "cartoon_set_test")
    test_images, test_labels = load_data(test_root_path)
    return test_images, test_labels


# Create TF Model.
class B2Model:
    # Set layers.
    def __init__(self, input_shape):
        self.model = Sequential(
            [
                # Convolution Layer with 32 filters and a kernel size of 5.
                Conv2D(
                    32, kernel_size=(5, 5), activation="relu", input_shape=input_shape
                ),
                # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
                MaxPooling2D(pool_size=(2, 2)),
                # Convolution Layer with 64 filters and a kernel size of 3.
                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
                MaxPooling2D(pool_size=(2, 2)),
                # Flatten the data to a 1-D vector for the fully connected layer.
                Flatten(),
                # Fully connected layer.
                Dense(1024, activation="relu"),
                # Apply Dropout.
                Dropout(0.5),
                # Output layer, class prediction.
                Dense(num_classes, activation="softmax"),
            ]
        )
        self.model.summary()
        adam = tf.keras.optimizers.Adam(
            learning_rate=tf.Variable(learning_rate),
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7),
        )
        adam.iterations
        adam.decay = tf.Variable(0.0)
        self.model.compile(
            optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def train(
        self, training_batches, validation_batches, epochs=15, verbose=1, plot=True
    ):
        history = self.model.fit(
            training_batches,
            steps_per_epoch=len(training_batches),
            epochs=epochs,
            validation_data=validation_batches,
            validation_steps=len(validation_batches),
            verbose=verbose,
        )
        if plot:
            plot_performance(
                history.history["accuracy"],
                history.history["val_accuracy"],
                history.history["loss"],
                history.history["val_loss"],
            )
        return history.history["accuracy"][-1], history.history["val_accuracy"][-1]

    def evaluate(self, test_batches, verbose=1):
        return self.model.evaluate(x=test_batches, verbose=verbose)

    def test(self, test_batches, verbose=1, confusion_mesh=True, class_labels="auto"):
        predictions = self.model.predict(
            x=test_batches, steps=len(test_batches), verbose=verbose
        )
        predictions = np.round(predictions)
        predicted_labels = [np.argmax(label) for label in predictions]
        predicted_labels = np.array(predicted_labels)
        true_labels = np.array(np.concatenate([y for x, y in test_batches], axis=0))
        if confusion_mesh:
            plot_confusion_matrix(class_labels, predicted_labels, true_labels)
        return accuracy_score(true_labels, predicted_labels)


def b2_main():
    # Build neural network model.
    input_shape = (500, 500, 3)
    # input_shape = np.array(np.reshape(training_data.element_spec[0].shape, (3, )))
    b2_model = B2Model(input_shape)
    print("model built")

    save_dir = os.path.join(parent_dir, "B2_weights.h5")
    if train_model:
        # Load training data, split into training and validation sets
        x_dev, y_dev = get_training_data(parent_dir)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_dev, y_dev, test_size=0.2, random_state=0
        )
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        # Use tf.data API to shuffle and batch data.
        training_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        training_batches = (
            training_data.shuffle(len(training_data)).batch(batch_size).prefetch(1)
        )

        validation_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_batches = (
            validation_data.shuffle(len(validation_data)).batch(batch_size).prefetch(1)
        )
        print("data prepared")

        acc_B2_train, acc_B2_valid = b2_model.train(
            training_batches, validation_batches, epochs=epochs, verbose=2, plot=True
        )
        print("model trained")
        if save_model:
            b2_model.model.save_weights(save_dir)
            print("model saved")
    else:  # no model training, load saved model
        b2_model.model.load_weights(save_dir)
        print("model loaded")

    # Load testing data
    x_test, y_test = get_testing_data(parent_dir)
    x_test_post, _, y_test_post, _ = train_test_split(
        x_test, y_test, test_size=0.001, random_state=0
    )
    x_test_post = np.array(x_test_post)
    y_test_post = np.array(y_test_post)
    print("testing data loaded")

    test_data = tf.data.Dataset.from_tensor_slices((x_test_post, y_test_post))
    test_batches = test_data.shuffle(len(test_data)).batch(batch_size).prefetch(1)
    print("testing data prepared")

    # Evaluate model on test set
    acc_B2_test = b2_model.test(test_batches, verbose=2, confusion_mesh=True)
    print("model tested, accuracy: ", acc_B2_test)

    # conv_net = ConvNet()
    # print("model built")

    # Stochastic gradient descent optimizer.
    # optimizer = tf.optimizers.Adam(learning_rate)

    # # Run training for the given number of steps.
    # for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    #     # Run the optimization to update W and b values.
    #     run_optimization(batch_x, batch_y, conv_net, optimizer)
    #     print(step)
    #     if step % display_step == 0:
    #         pred = conv_net(batch_x)
    #         loss = cross_entropy_loss(pred, batch_y)
    #         acc = accuracy(pred, batch_y)
    #         print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

    # # Test model on validation set.
    # pred = conv_net(x_test)
    # print("Test Accuracy: %f" % accuracy(pred, y_test))


if __name__ == "__main__":
    b2_main()
