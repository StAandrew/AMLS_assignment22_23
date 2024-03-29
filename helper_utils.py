""" Helper functions.

This file contains all paths and helper functions used in the project.
"""

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.io import read_file, decode_png
from tensorflow.image import resize
import logging
import sys


# Specify root direcries
root_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(root_dir, "Datasets")

# Specify celeba directories
celeba_train_img_dir = os.path.join(datasets_dir, "celeba", "img")
celeba_train_label_path = os.path.join(datasets_dir, "celeba", "labels.csv")
celeba_test_img_dir = os.path.join(datasets_dir, "celeba_test", "img")
celeba_test_label_path = os.path.join(datasets_dir, "celeba_test", "labels.csv")

# Specify extra directories
celeba_features_train_path = os.path.join(datasets_dir, "celeba", "features.npz")
celeba_features_test_path = os.path.join(datasets_dir, "celeba_test", "features.npz")

a1_model_path = os.path.join(root_dir, "A1", "a1_model.pkl")
a1_figure_learning_path = os.path.join(root_dir, "A1", "a1_learning_curve.png")
a1_figure_learning_file_path = os.path.join(
    root_dir, "A1", "a1_learning_curve_data.npz"
)
a1_figure_confusion_matrix_path = os.path.join(
    root_dir, "A1", "a1_confusion_matrix.png"
)
a1_figure_c_performance_path = os.path.join(root_dir, "A1", "a1_c_performance.png")
a1_figure_gamma_performance_path = os.path.join(
    root_dir, "A1", "a1_gamma_performance.png"
)

a2_model_path = os.path.join(root_dir, "A2", "a2_model.pkl")
a2_figure_learning_path = os.path.join(root_dir, "A2", "a2_learning_curve.png")
a2_figure_learning_file_path = os.path.join(
    root_dir, "A2", "a2_learning_curve_data.npz"
)
a2_figure_confusion_matrix_path = os.path.join(
    root_dir, "A2", "a2_confusion_matrix.png"
)
a2_figure_c_performance_path = os.path.join(root_dir, "A2", "a2_c_performance.png")
a2_figure_gamma_performance_path = os.path.join(
    root_dir, "A2", "a2_gamma_performance.png"
)

# Specify default cartoon directories
cartoon_img_dir = os.path.join(datasets_dir, "cartoon_set", "img")
cartoon_label_path = os.path.join(datasets_dir, "cartoon_set", "labels.csv")
cartoon_test_img_dir = os.path.join(datasets_dir, "cartoon_set_test", "img")
cartoon_test_label_path = os.path.join(datasets_dir, "cartoon_set_test", "labels.csv")

cartoon_features_dir = os.path.join(datasets_dir, "cartoon_set", "features.npz")
cartoon_eyes_dir = os.path.join(datasets_dir, "cartoon_set", "eyes")
cartoon_eyes_label_path = os.path.join(datasets_dir, "cartoon_set", "eyes_labels.csv")
cartoon_jaws_dir = os.path.join(datasets_dir, "cartoon_set", "jaws")
cartoon_jaws_label_path = os.path.join(datasets_dir, "cartoon_set", "jaws_labels.csv")

cartoon_test_features_dir = os.path.join(
    datasets_dir, "cartoon_set_test", "features.npz"
)
cartoon_test_eyes_dir = os.path.join(datasets_dir, "cartoon_set_test", "eyes")
cartoon_test_eyes_label_path = os.path.join(
    datasets_dir, "cartoon_set_test", "eyes_labels.csv"
)
cartoon_test_jaws_dir = os.path.join(datasets_dir, "cartoon_set_test", "jaws")
cartoon_test_jaws_label_path = os.path.join(
    datasets_dir, "cartoon_set_test", "jaws_labels.csv"
)

b1_model_path = os.path.join(root_dir, "B1", "B1_weights.h5")
b2_model_path = os.path.join(root_dir, "B2", "B2_weights.h5")

b1_root_dir = os.path.join(root_dir, "B1")
b2_root_dir = os.path.join(root_dir, "B2")


def initial_config(warnings_off=True, cpu_only_training=False):
    """
    Initial configuration for the project.
    :param warnings_off: Disable tensorflow warnings
    :param cpu_only_training: Disable GPU usage
    :return: None
    """
    # Disable tensorflow warnings
    if warnings_off:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Setup logging
    logging.basicConfig(
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Check if GPU(s) are available, limit memory growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if len(gpus) > 0:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, False)
            except RuntimeError as e:
                logger.error(e)
    elif cpu_only_training:
        logger.warning("No GPU found. Attempting to train on CPU...\n")
    else:
        raise Exception(
            "No GPU found. Quitting...\n You can change this behaviour by setting the cpu_only_training flag to True."
        )

    try:
        paths = []
        paths.append(cartoon_eyes_dir)
        paths.append(cartoon_test_eyes_dir)
        paths.append(cartoon_jaws_dir)
        paths.append(cartoon_test_jaws_dir)

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info("Created additional directory: " + path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise Exception(f"Error creating directories: {e}")
            exit(1)
    return logger


def helper_plot_learning(figure_save_path, train_sizes, train_scores, val_scores):
    plt.figure(figsize=(11, 8), dpi=80)
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")
    plt.title("SVM learning curve", fontsize=30)
    plt.xlabel("Number of Training Samples", fontsize=20)
    plt.ylabel("Accuracy Score, %", fontsize=20)
    plt.legend()
    plt.savefig(figure_save_path, dpi=300)


def helper_plot_confusion_matrix(figure_save_path, confusion_matrix):
    total_elements = sum(map(sum, confusion_matrix))
    plt.figure(figsize=(8, 6), dpi=80)
    sn.heatmap(
        confusion_matrix,
        cmap="Blues",
        fmt="d",
        annot_kws={"size": 20, "color": "red"},
        cbar_kws={"ticks": [0, 1], "label": "Number of elements laballed as such"},
        annot=True,
    )
    plt.xlabel("Predicted Label", fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    plt.title(
        f"SVM model confusion matrix\nTotal: {total_elements} elements", fontsize=16
    )
    plt.savefig(figure_save_path, dpi=300)


def helper_plot_grid_c(figure_save_path, C, mean_scores):
    plt.figure(figsize=(12, 8), dpi=80)
    # plt.rcParams["figure.figsize"] = [15, 10]
    plt.scatter(C, mean_scores, label="Test Score vs C", s=80, alpha=1)
    plt.grid(which="both", linestyle=":", linewidth=1.5)
    plt.xlabel("SVM parameter C", fontsize=20)
    plt.ylabel("Accuracy Score, %", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.title("SVM model, linear kernel. Impact of C on performance", fontsize=20)
    plt.savefig(figure_save_path, dpi=300)


def helper_plot_grid_gamma(figure_save_path, gamma, mean_scores):
    plt.figure(figsize=(15, 10), dpi=80)
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.scatter(gamma, mean_scores, label="Test Score vs C", s=80, alpha=1)
    plt.grid(which="both", linestyle=":", linewidth=1.5)
    plt.xlabel("SVM parameter gamma", fontsize=20)
    plt.ylabel("Accuracy Score, %", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.title("SVM model, linear kernel. Impact of Gamma on performance", fontsize=20)
    plt.savefig(figure_save_path, dpi=300)


# etc...
def plot_performance(save_path, accuracy, val_accuracy, loss, val_loss, title=None):
    """
    Plots the history of the training phase and validation phase. It compares in two different subplots the accuracy
    and the loss of the model.
    :param accuracy: list of values for every epoch.
    :param val_accuracy: list of values for every epoch.
    :param loss: list of values for every epoch.
    :param val_loss: list of values for every epoch.
    :param title: tile of the figure printed. default_value=None
    :return:
    """
    x_axis = [i for i in range(1, len(accuracy) + 1)]
    sn.set()
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    # First subplot
    plt.subplot(211)
    plt.plot(x_axis, accuracy)
    plt.plot(x_axis, val_accuracy)
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Valid"], loc="lower right")
    # Second subplot
    plt.subplot(212)
    plt.plot(x_axis, loss)
    plt.plot(x_axis, val_loss)
    plt.ylabel("Loss")
    plt.ylim(top=0.7)
    plt.xlabel("Epoch")
    # Legend
    plt.legend(["Train", "Valid"], loc="upper right")
    plt.savefig(save_path, dpi=600)


def plot_confusion_matrix(
    logger, save_path, class_labels, predicted_labels, true_labels, title=None
):
    """
    Plots the confusion matrix given both the true and predicted results.
    :param class_labels: list of the names of the labels.
    :param predicted_labels: list of the predicted labels.
    :param true_labels: list of the true labels.
    :param title: tile of the figure printed
    :return:
    """
    sn.set()
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    confusion_grid = pd.crosstab(true_labels, predicted_labels, normalize=True)
    # Generate a custom diverging colormap
    color_map = sn.diverging_palette(355, 250, as_cmap=True)
    sn.heatmap(
        confusion_grid,
        cmap=color_map,
        vmax=0.5,
        vmin=0,
        center=0,
        xticklabels=class_labels,
        yticklabels=class_labels,
        square=True,
        linewidths=2,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(save_path, dpi=600)
    # Print a detailed report on the classification results
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(true_labels, predicted_labels)}")
