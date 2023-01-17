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

# Specify basic directories
celeba_train_img_dir = os.path.join(datasets_dir, "celeba", "img")
celeba_train_label_dir = os.path.join(datasets_dir, "celeba", "labels.csv")
celeba_test_img_dir = os.path.join(datasets_dir, "celeba_test", "img")
celeba_test_label_dir = os.path.join(datasets_dir, "celeba_test", "labels.csv")

cartoon_train_img_dir = os.path.join(datasets_dir, "cartoon_set", "img")
cartoon_train_label_dir = os.path.join(datasets_dir, "cartoon_set_test", "labels.csv")
cartoon_test_img_dir = os.path.join(datasets_dir, "cartoon_set", "img")
cartoon_test_label_dir = os.path.join(datasets_dir, "cartoon_set_test", "labels.csv")

# Specify extra directories
celeba_smile_dir = os.path.join(datasets_dir, "celeba_smile")
celeba_train_img_smile_dir = os.path.join(datasets_dir, "celeba_smile", "img")
celeba_train_label_smile_dir = os.path.join(datasets_dir, "celeba_smile", "labels.csv")
celeba_test_img_smile_dir = os.path.join(datasets_dir, "celeba_test_smile", "img")
celeba_test_label_smile_dir = os.path.join(
    datasets_dir, "celeba_test_smile", "labels.csv"
)

cartoon_no_glasses_dir = os.path.join(datasets_dir, "cartoon_set_no_glasses")
cartoon_train_img_no_glasses_dir = os.path.join(
    datasets_dir, "cartoon_set_no_glasses", "img"
)
cartoon_train_label_no_glasses_dir = os.path.join(
    datasets_dir, "cartoon_set_no_glasses", "labels.csv"
)
cartoon_test_img_no_glasses_dir = os.path.join(
    datasets_dir, "cartoon_set_test_no_glasses", "img"
)
cartoon_test_label_no_glasses_dir = os.path.join(
    datasets_dir, "cartoon_set_test_no_glasses", "labels.csv"
)


def initial_config(warnings_off=True, cpu_only_training=False):
    """
    Initial configuration for the project.
    :param warnings_off: Disable tensorflow warnings
    :param cpu_only_training: Disable GPU usage
    :return: None
    """
    # Disable tensorflow warnings
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
                tf.config.experimental.set_memory_growth(gpu, True)
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
        paths.append(celeba_smile_dir)
        paths.append(cartoon_no_glasses_dir)
        paths.append(celeba_train_img_smile_dir)
        paths.append(celeba_train_label_smile_dir)
        paths.append(celeba_test_img_smile_dir)
        paths.append(celeba_test_label_smile_dir)
        paths.append(cartoon_train_img_no_glasses_dir)
        paths.append(cartoon_train_label_no_glasses_dir)
        paths.append(cartoon_test_img_no_glasses_dir)
        paths.append(cartoon_test_label_no_glasses_dir)

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise Exception(f"Error creating directories: {e}")

