from helper_utils import *
from data_processing import *
from sklearn import svm
from keras.callbacks import EarlyStopping

from A1.a1 import A1
# from A2.a2 import A2
from B1.b1 import B1
# from B2.b2 import B2

logger = initial_config()
run_a1 = False
run_a2 = False
run_b1 = True
run_b2 = False

if run_a1 or run_a2:
    logger.info("Loading CelebA dataset features...")
    # Prepare training and verification data for A1 & A2
    celeba_features, labels_df = load_features(celeba_features_train_dir, celeba_train_label_dir)
    if celeba_features is None or labels_df is None:
        logger.info("Celeba features not found, loading raw data and extracting features...")
        images, labels_df = load_datasets(celeba_train_img_dir, celeba_train_label_dir, "img_name", "gender", "smiling")
        celeba_features = extract_face_features(images)
        save_dataset(celeba_features, celeba_features_train_dir)

    logger.info("Loading CelebA test dataset features...")
    celeba_features_test, labels_df_test = load_features(celeba_features_test_dir, celeba_test_label_dir)
    if celeba_features_test is None or labels_df_test is None:
        logger.info("Celeba test features not found, loading raw data and extracting features...")
        images, labels_df_test = load_datasets(celeba_test_img_dir, celeba_test_label_dir, "img_name", "gender", "smiling")
        celeba_features_test = extract_face_features(images)
        save_dataset(celeba_features_test, celeba_features_test_dir)


if run_a1:
    # A1
    logger.info("Running A1: Gender Classification")
    test_size = 0 # because grid search will be used
    label_name = "gender"

    jawline_arr = extract_jawline_features(celeba_features)
    jawline_data_train, _, jawline_label_train, _ = shuffle_split(jawline_arr, labels_df, "img_name", "gender", "smiling", label_name, test_size, logger)
    jawline_data_train = data_reshape(jawline_data_train)

    jawline_arr_test = extract_jawline_features(celeba_features_test)
    jawline_data_test, _, jawline_label_test, _ = shuffle_split(jawline_arr_test, labels_df_test, "img_name", "gender", "smiling", label_name, 0, logger)
    jawline_data_test = data_reshape(jawline_data_test)

    model = svm.SVC()
    a1 = A1(jawline_data_train, jawline_label_train, jawline_data_test, jawline_label_test, logger)
    logger.info("Loading A1 model...")
    if not a1.load_model(a1_model_path):
        logger.info("A1 model not found, training using grid search...")
        a1.train_grid_fit(model)
        a1.save_model(a1_model_path)
    a1.evaluate_best_model()
    a1.output_info()
    a1.plot_learning(a1_figure_learning_path, a1_figure_learning_file_path)
    a1.plot_confusion_matrix(a1_figure_confusion_matrix_path)
    a1.plot_grid_c(a1_figure_c_performance_path)
    a1.plot_grid_gamma(a1_figure_gamma_performance_path)


if run_a2:
    # A2
    logger.info("Running A2: Smile detection")
    test_size = 0 # because grid search will be used
    label_name = "smiling"

    smile_arr = extract_smile_features(celeba_features)
    smile_data_train, _, smile_label_train, _ = shuffle_split(smile_arr, labels_df, "img_name", "gender", "smiling", label_name, test_size, logger)
    smile_data_train = data_reshape(smile_data_train)

    smile_arr_test = extract_smile_features(celeba_features_test)
    smile_data_test, _, smile_label_test, _ = shuffle_split(smile_arr_test, labels_df_test, "img_name", "gender", "smiling", label_name, 0, logger)
    smile_data_test = data_reshape(smile_data_test)

    model = svm.SVC()
    a2 = A1(smile_data_train, smile_label_train, smile_data_test, smile_label_test, logger)
    logger.info("Loading A2 model...")
    if not a2.load_model(a2_model_path):
        logger.info("A2 model not found, training using grid search...")
        a2.train_grid_fit(model)
        a2.save_model(a2_model_path)
    a2.evaluate_best_model()
    a2.output_info()
    a2.plot_learning(a2_figure_learning_path, a2_figure_learning_file_path)
    a2.plot_confusion_matrix(a2_figure_confusion_matrix_path)
    a2.plot_grid_c(a2_figure_c_performance_path)
    a2.plot_grid_gamma(a2_figure_gamma_performance_path)


if run_b1:
    img_size = (250, 250)
    validation_size = 0.2
    batch_size = 8  # reduce if not enough GPU vRAM available
    epochs = 10
    filename_column = "file_name"
    label_name = "face_shape"

    logger.info("Loading resized gray Cartoon dataset images...")
    cartoon_gray_images, cartoon_labels_df = load_datasets(cartoon_resized_gray_images_path, cartoon_train_label_dir, "file_name", "eye_color", "face_shape")
    if cartoon_gray_images is None:
        logger.info("Cartoon resized gray images not found. Loading raw images...")
        gray_images, _ = load_datasets(cartoon_train_img_dir, cartoon_train_label_dir, "file_name", "eye_color", "face_shape")
        logger.info("Resizing gray images...")
        cartoon_gray_images = crop_resize_images_func(img_size, gray_images)
        logger.info("Saving gray images...")
        save_resized_images(cartoon_gray_images, cartoon_resized_gray_images_path)
        del gray_images

    logger.info("Loading resized gray Cartoon test dataset images...")
    cartoon_test_gray_images, cartoon_test_labels_df = load_datasets(cartoon_test_resized_gray_images_path, cartoon_test_label_dir, "file_name", "eye_color", "face_shape")
    if cartoon_test_gray_images is None:
        logger.info("Cartoon resized gray images not found. Loading gray test images...")
        gray_images, _ = load_datasets(cartoon_test_img_dir, cartoon_test_label_dir, "file_name", "eye_color", "face_shape")
        logger.info("Resizing gray test images...")
        cartoon_test_gray_images = crop_resize_images_func(img_size, gray_images)
        logger.info("Saving gray test images...")
        save_resized_images(cartoon_test_gray_images, cartoon_test_resized_gray_images_path)
        del gray_images
    
    del cartoon_gray_images, cartoon_test_gray_images, cartoon_labels_df, cartoon_test_labels_df


    cartoon_train_batches, cartoon_validation_batches = data_split_preparation(cartoon_resized_gray_images_path, cartoon_train_label_dir, filename_column, label_name, img_size, validation_size, batch_size)
    cartoon_test_batches = data_preparation(cartoon_test_resized_gray_images_path, cartoon_test_label_dir, filename_column, label_name, img_size, batch_size)

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss", restore_best_weights=True, patience=5, verbose=1
    # )

    input_shape = cartoon_train_batches.image_shape
    logger.debug(f"input shape: {input_shape}")
    model = B1(input_shape)
    acc_B1_train, acc_B1_valid = model.train(
        cartoon_train_batches, cartoon_validation_batches, epochs=epochs, verbose=2, plot=True
    )
    logger.info(f"Training accuracy: {str(acc_B1_train)}")
    # model.model.save_weights(b1_model_path)
    acc_B1_test = model.test(logger, cartoon_test_batches, verbose=2, confusion_mesh=True)
    logger.info(f"model tested, accuracy: {str(acc_B1_test)}")


if run_b2:
    logger.info("Loading resized Cartoon dataset images...")
    cartoon_images, cartoon_labels_df = load_datasets(cartoon_resized_images_path, cartoon_train_label_dir, "file_name", "eye_color", "face_shape", grayscale=False)
    if cartoon_images is None:
        logger.info("Cartoon resized images not found. Loading raw data...")
        images, cartoon_labels_df = load_datasets(cartoon_train_img_dir, cartoon_train_label_dir, "file_name", "eye_color", "face_shape", grayscale=False)
        logger.info("Resizing RGB images...")
        cartoon_images = crop_resize_images_func(images, grayscale=False)
        logger.debug("loaded images: {}".format(cartoon_images.shape))
        logger.info("Saving RGB images...")
        save_resized_images(cartoon_images, cartoon_resized_images_path, grayscale=False)
        del images
    
    logger.info("Loading resized Cartoon test dataset images...")
    cartoon_test_images, cartoon_test_labels_df = load_datasets(cartoon_test_resized_images_path, cartoon_test_label_dir, "file_name", "eye_color", "face_shape", grayscale=False)
    if cartoon_test_images is None:
        logger.info("Cartoon test resized images not found. Loading raw data...")
        images, cartoon_test_labels_df = load_datasets(cartoon_test_img_dir, cartoon_test_label_dir, "file_name", "eye_color", "face_shape", grayscale=False)
        logger.info("Resizing RGB test images...")
        cartoon_test_images = crop_resize_images_func(images, grayscale=False)
        logger.info("Saving RGB test images...")
        save_resized_images(cartoon_test_images, cartoon_test_resized_images_path, grayscale=False)
        del images

    # B2
    # Prepare RGB pirctures for B2
    cartoon_images, _ = load_datasets(cartoon_train_img_dir, cartoon_train_label_dir, "file_name", "eye_color", "face_shape", grayscale=False)

    # labels, images = load_datasets(cartoon_train_img_dir, cartoon_train_label_dir, "eye_color", "file_name", grayscale=False)
    # test_labels, test_images = load_datasets(cartoon_test_img_dir, cartoon_test_label_dir, "eye_color", "file_name", grayscale=False)

logger.info("Finished.")

# logger.info("Loading Cartoon dataset features...")
# cartoon_features, _ = load_features(cartoon_train_features_dir, cartoon_train_label_dir)
# if cartoon_features is None:
#     logger.info("Cartoon features not found. Extracting features...")
#     cartoon_features = extract_face_features(cartoon_gray_images)
#     logger.info("Saving features...")
#     save_dataset(cartoon_features, cartoon_train_features_dir)
# try:
#     del gray_images
# except:
#     pass

# logger.info("Loading Cartoon test dataset features...")
# cartoon_test_features, _ = load_features(cartoon_test_features_dir, cartoon_test_label_dir)
# if cartoon_test_features is None:
#     logger.info("Cartoon features not found. Extracting test features...")
#     cartoon_test_features = extract_face_features(cartoon_test_gray_images)
#     logger.info("Saving test features...")
#     save_dataset(cartoon_test_features, cartoon_test_features_dir)
# try:
#     del gray_images
# except:
#     pass