paths = dict(
    train_data = "dataset/train",   # the base path of the training images
    test_data = "dataset/test",     # the base path of the validation images
    val_data = "dataset/validation", # the base path of the validation images
    label_excel = "dataset/labelnames.csv", # the base of the excel where the labels of the excels are kept
    inference_img_path = "dataset/test/adidas/aadidas_ (16).jpg",    # the image on which the prediction will be done
    pre_trained_model_path = "checkpoints/efficientnet_best_iter1.h5", # path of the pre-trained model if used
    inference_model_path = "checkpoints/ensemble_model_CNN_test.h5",   # the path where model inference will ke kept
)

train_params = dict(
    batch_size = 2, # batch size fot model
    epochs = 10,   # number of epoch the model will run
)

options = dict(
    train_mode = "ensemble",     # choose one of the following four : simple_CNN/resnet/efficientnet/ensemble
    pre_trained = False,    # defined whether to use a pre-trained model or not
    logs_name = "CNN_test",     # the name under which logs will be kept
    increase_train_image = False
)