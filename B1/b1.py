from helper_utils import *
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


# MNIST dataset parameters.
num_classes = 5  # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.001
training_steps = 78
batch_size = 128
display_step = 10
epochs = 10
num_samples = 100  # -1 for all available samples
validation_split = 0.2

# Network parameters.
conv1_filters = 32  # number of filters for 1st conv layer.
conv2_filters = 64  # number of filters for 2nd conv layer.
fc1_units = 1024  # number of neurons for 1st fully-connected layer.


# Create TF Model.
class B1:
    # Set layers.
    def __init__(self, input_shape):
        self.model = Sequential(
            [
                # Convolution Layer with 32 filters and a kernel size of 5.
                # Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=input_shape),
                Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=input_shape, padding="same"),
                # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
                MaxPooling2D(pool_size=(2, 2), strides=2),
                # Convolution Layer with 64 filters and a kernel size of 3.
                Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
                # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
                MaxPooling2D(pool_size=(2, 2), strides=2),
                # Convolution Layer with 64 filters and a kernel size of 3.
                Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
                # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
                MaxPooling2D(pool_size=(2, 2), strides=2),
                # Flatten the data to a 1-D vector for the fully connected layer.
                Flatten(),
                # Fully connected layer.
                # Dense(1024, activation="relu"),
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
            optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"]
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

    def test(self, logger, test_batches, verbose=1, confusion_mesh=True, class_labels="auto"):
        predictions = self.model.predict(
            x=test_batches, steps=len(test_batches), verbose=verbose
        )
        predictions = np.round(predictions)
        predicted_labels = np.array(np.argmax(predictions, axis=-1))
        true_labels = np.array(test_batches.classes)
        if confusion_mesh:
            plot_confusion_matrix(logger, class_labels, predicted_labels, true_labels)
        return accuracy_score(true_labels, predicted_labels)
