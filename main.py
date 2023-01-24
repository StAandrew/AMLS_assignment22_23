from helper_utils import *
from data_processing import *
from sklearn import svm
from keras.callbacks import EarlyStopping

from A1.a1 import A1
# from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

logger = initial_config()
run_a1 = False
run_a2 = False
run_b1_images = True
run_b1_features = False
run_b2 = False

if run_a1 or run_a2:
    logger.info("Loading CelebA dataset features...")
    # Prepare training and verification data for A1 & A2
    celeba_features, label_df = load_features(celeba_features_train_path, celeba_train_label_path)
    if celeba_features is None or label_df is None:
        logger.info("Celeba features not found, loading raw data and extracting features...")
        images, label_df = load_datasets(celeba_train_img_dir, celeba_train_label_path, "img_name", "gender", "smiling")
        celeba_features = extract_face_features(images)
        save_dataset(celeba_features, celeba_features_train_path)

    logger.info("Loading CelebA test dataset features...")
    celeba_features_test, label_df_test = load_features(celeba_features_test_path, celeba_test_label_path)
    if celeba_features_test is None or label_df_test is None:
        logger.info("Celeba test features not found, loading raw data and extracting features...")
        images, label_df_test = load_datasets(celeba_test_img_dir, celeba_test_label_path, "img_name", "gender", "smiling")
        celeba_features_test = extract_face_features(images)
        save_dataset(celeba_features_test, celeba_features_test_path)


if run_a1:
    # A1
    logger.info("Running A1: Gender Classification")
    test_size = 0 # because grid search will be used
    label_name = "gender"

    jawline_arr = extract_jawline_features(celeba_features)
    jawline_data_train, _, jawline_label_train, _ = shuffle_split(jawline_arr, label_df, "img_name", "gender", "smiling", label_name, test_size, logger)
    jawline_data_train = data_reshape(jawline_data_train)

    jawline_arr_test = extract_jawline_features(celeba_features_test)
    jawline_data_test, _, jawline_label_test, _ = shuffle_split(jawline_arr_test, label_df_test, "img_name", "gender", "smiling", label_name, 0, logger)
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
    smile_data_train, _, smile_label_train, _ = shuffle_split(smile_arr, label_df, "img_name", "gender", "smiling", label_name, test_size, logger)
    smile_data_train = data_reshape(smile_data_train)

    smile_arr_test = extract_smile_features(celeba_features_test)
    smile_data_test, _, smile_label_test, _ = shuffle_split(smile_arr_test, label_df_test, "img_name", "gender", "smiling", label_name, 0, logger)
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


if run_b1_images:
    filename_column = "file_name"
    label_name = "face_shape"
    jaw_rectangle = (260, 390, 155, 345)
    black_rectangles = [(245, 280, 225, 275), (260, 310, 185, 315)]
    # black_rectangles = []
    reference_colour_position = (320, 249)
    check_colour_position = (385, 278)
    img_size = (jaw_rectangle[1]-jaw_rectangle[0], jaw_rectangle[3]-jaw_rectangle[2])
    validation_size = 0.2
    batch_size = 16  # reduce if not enough GPU vRAM available
    epochs = 10

    logger.info("Loading resized Cartoon dataset images...")
    if not check_if_dataset_present(cartoon_jaws_dir, cartoon_jaws_label_path, filename_column):
        logger.info("Cartoon jaw images not found. Loading raw data...")
        images, cartoon_label_df = load_datasets(cartoon_img_dir, cartoon_label_path, filename_column, "eye_color", "face_shape")
        logger.info("Resizing and removing images with beards...")
        cartoon_jaw_images, cartoon_jaw_df = extract_jaw_rectangle_remove_undetectable(logger, images, cartoon_label_df, jaw_rectangle, black_rectangles, reference_colour_position, check_colour_position)
        logger.info("Saving cartoon jaw images...")
        save_resized_images(cartoon_jaw_images, cartoon_jaws_dir)
        save_dataframe(logger, cartoon_jaw_df, cartoon_jaws_label_path)
        del images
    
    logger.info("Loading resized Cartoon test dataset images...")
    if not check_if_dataset_present(cartoon_test_jaws_dir, cartoon_test_jaws_label_path, filename_column):
        logger.info("Cartoon test resized images not found. Loading raw data...")
        images, cartoon_test_label_df = load_datasets(cartoon_test_img_dir, cartoon_test_label_path, filename_column, "eye_color", "face_shape")
        logger.info("Resizing and removing images with beards...")
        cartoon_test_jaw_images, cartoon_test_jaw_df = extract_jaw_rectangle_remove_undetectable(logger, images, cartoon_test_label_df, jaw_rectangle, black_rectangles, reference_colour_position, check_colour_position)
        logger.info("Saving cartoon test jaw images...")
        save_resized_images(cartoon_test_jaw_images, cartoon_test_jaws_dir)
        save_dataframe(logger, cartoon_test_jaw_df, cartoon_test_jaws_label_path)
        del images

    cartoon_train_batches, cartoon_validation_batches = data_split_preparation(cartoon_jaws_dir, cartoon_jaws_label_path, filename_column, label_name, img_size, validation_size, batch_size)
    cartoon_test_batches = data_preparation(cartoon_test_jaws_dir, cartoon_test_jaws_label_path, filename_column, label_name, img_size, batch_size)

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


if run_b1_features:
    filename_column = "file_name"
    label_name = "face_shape"
    validation_size = 0.2
    batch_size = 16  # reduce if not enough GPU vRAM available
    epochs = 10

    if check_if_dataset_present(cartoon_features_dir, cartoon_label_path, filename_column):
        logger.info("Cartoon features found. Loading...")
        cartoon_features, cartoon_label_df = load_datasets(cartoon_features_dir, cartoon_label_path, filename_column, label_name, label_name)
    else:
        logger.info("Cartoon features not found. Loading raw images...")
        images, cartoon_label_df = load_datasets(cartoon_img_dir, cartoon_label_path, filename_column, "eye_color", "face_shape", grayscale=True)
        logger.info("Extracting features...")
        cartoon_features = extract_face_features(images)
        logger.info("Saving features...")
        save_dataset(cartoon_features, cartoon_features_dir)
        del images
        # del cartoon_features
    
    if check_if_dataset_present(cartoon_test_features_dir, cartoon_test_label_path, filename_column):
        logger.info("Cartoon test features found. Loading...")
        cartoon_test_features, cartoon_test_label_df = load_datasets(cartoon_test_features_dir, cartoon_test_label_path, filename_column, label_name, label_name)
    else:
        logger.info("Cartoon test features not found. Loading raw images...")
        images, cartoon_test_label_df = load_datasets(cartoon_test_img_dir, cartoon_test_label_path, filename_column, "eye_color", "face_shape", grayscale=True)
        logger.info("Extracting features...")
        cartoon_test_features = extract_face_features(images)
        logger.info("Saving features...")
        save_dataset(cartoon_test_features, cartoon_test_features_dir)
        del images
        # del cartoon_test_features
    
    jawline_arr = extract_jawline_features(cartoon_features)
    jawline_data_train, _, jawline_label_train, _ = shuffle_split(jawline_arr, cartoon_label_df, filename_column, "eye_color", "face_shape", label_name, validation_size, logger)
    jawline_data_train = data_reshape(jawline_data_train)

    jawline_arr_test = extract_jawline_features(cartoon_test_features)
    jawline_data_test, _, jawline_label_test, _ = shuffle_split(jawline_arr_test, cartoon_test_label_df, filename_column, "eye_color", "face_shape", label_name, 0, logger)
    jawline_data_test = data_reshape(jawline_data_test)
    
    model = svm.SVC()
    b1 = A1(jawline_data_train, jawline_label_train, jawline_data_test, jawline_label_test, logger)
    logger.info("Loading B1 SVM model...")
    if not b1.load_model(b1_model_path):
        logger.info("B1 model not found, training using grid search...")
        b1.train_grid_fit(model)
        b1.save_model(b1_model_path)
    b1.evaluate_best_model()
    b1.output_info()
    b1.plot_learning(b1_figure_learning_path, b1_figure_learning_file_path)
    b1.plot_confusion_matrix(b1_figure_confusion_matrix_path)
    b1.plot_grid_c(b1_figure_c_performance_path)
    b1.plot_grid_gamma(b1_figure_gamma_performance_path)

    exit()

    img_size = ()
    cartoon_train_batches, cartoon_validation_batches = data_split_preparation(cartoon_features_dir, cartoon_label_path, filename_column, label_name, img_size, validation_size, batch_size)
    cartoon_test_batches = data_preparation(cartoon_test_features_dir, cartoon_test_label_path, filename_column, label_name, img_size, batch_size)

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
    filename_column = "file_name"
    label_name = "eye_color"
    eye_rectangle = (248, 275, 190, 310)
    black_rectangle = (245, 280, 225, 275)
    img_size = (eye_rectangle[1]-eye_rectangle[0], eye_rectangle[3]-eye_rectangle[2])
    validation_size = 0.2
    batch_size = 16  # reduce if not enough GPU vRAM available
    epochs = 10

    logger.info("Loading resized Cartoon dataset images...")
    if not check_if_dataset_present(cartoon_eyes_dir, cartoon_eyes_label_path, filename_column):
        logger.info("Cartoon eye images not found. Loading raw data...")
        images, cartoon_label_df = load_datasets(cartoon_img_dir, cartoon_label_path, filename_column, "eye_color", "face_shape", grayscale=False)
        logger.info("Resizing and removing images with glasses...")
        cartoon_eye_images, cartoon_eye_df = extract_eye_rectangle_remove_glasses(logger, images, cartoon_label_df, eye_rectangle, black_rectangle, grayscale=False)
        logger.info("Saving cartoon eye images...")
        save_resized_images(cartoon_eye_images, cartoon_eyes_dir, grayscale=False)
        save_dataframe(logger, cartoon_eye_df, cartoon_eyes_label_path)
        del images
    
    logger.info("Loading resized Cartoon test dataset images...")
    if not check_if_dataset_present(cartoon_test_eyes_dir, cartoon_test_eyes_label_path, filename_column):
        logger.info("Cartoon test resized images not found. Loading raw data...")
        images, cartoon_test_label_df = load_datasets(cartoon_test_img_dir, cartoon_test_label_path, filename_column, "eye_color", "face_shape", grayscale=False)
        logger.info("Resizing and removing images with glasses...")
        cartoon_test_eye_images, cartoon_test_eye_df = extract_eye_rectangle_remove_glasses(logger, images, cartoon_test_label_df, eye_rectangle, black_rectangle, grayscale=False)
        logger.info("Saving cartoon test eye images...")
        save_resized_images(cartoon_test_eye_images, cartoon_test_eyes_dir, grayscale=False)
        save_dataframe(logger, cartoon_test_eye_df, cartoon_test_eyes_label_path)
        del images

    cartoon_train_batches, cartoon_validation_batches = data_split_preparation(cartoon_eyes_dir, cartoon_eyes_label_path, filename_column, label_name, img_size, validation_size, batch_size)
    cartoon_test_batches = data_preparation(cartoon_test_eyes_dir, cartoon_test_eyes_label_path, filename_column, label_name, img_size, batch_size)
    
    input_shape = cartoon_train_batches.image_shape
    logger.debug(f"input shape: {input_shape}")
    model = B2(input_shape)
    acc_B2_train, acc_B2_valid = model.train(
        cartoon_train_batches, cartoon_validation_batches, epochs=epochs, verbose=2, plot=True
    )
    logger.info(f"Training accuracy: {str(acc_B2_train)}")
    # model.model.save_weights(b2_model_path)
    acc_B2_test = model.test(logger, cartoon_test_batches, verbose=2, confusion_mesh=True)
    logger.info(f"model tested, accuracy: {str(acc_B2_test)}")
logger.info("Finished.")

# logger.info("Loading Cartoon dataset features...")
# cartoon_features, _ = load_features(cartoon_train_features_dir, cartoon_train_label_path)
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
# cartoon_test_features, _ = load_features(cartoon_test_features_dir, cartoon_test_label_path)
# if cartoon_test_features is None:
#     logger.info("Cartoon features not found. Extracting test features...")
#     cartoon_test_features = extract_face_features(cartoon_test_gray_images)
#     logger.info("Saving test features...")
#     save_dataset(cartoon_test_features, cartoon_test_features_dir)
# try:
#     del gray_images
# except:
#     pass