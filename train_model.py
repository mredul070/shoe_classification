import os 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0, ResNet50

import config
from data_preprocess import transform_data, load_label_from_excel

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def simple_CNN_model():
    """This function defines a simple CNN model
    Returns:
        tf.keras.model: return the defined model
    """ 
    # initiate a sequential model
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics='accuracy')
    # print model summary
    model.summary()

    return model


def efficientnet():
    """This function defines the backbone of a efficientnet model
    Returns:
        tf.keras.model: return the defined model
    """ 
    # define input layer
    inputs = layers.Input(shape=(224, 224, 3))
    # add augmentation layer
    # x = img_augmentation(inputs)
    # define efficientnet config
    outputs = EfficientNetB0(include_top=True, weights=None, classes=2)(inputs)
    # generate the effcienet model
    model = tf.keras.Model(inputs, outputs)
    # compile the model
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    # print model summary
    model.summary()

    return model



def resnet():
    """This function defines the backbone of a resnet model
    Returns:
        tf.keras.model: return the defined model
    """    
    eff_model = ResNet50(
        include_top=False, 
        input_shape=(224, 224, 3),
        classes=2
    )

    for layer in eff_model.layers[:-1]:
        layer.trainable = False
    
    x = eff_model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)

    predictions = Dense(2, activation="softmax")(x)

    # creating the final model 
    model = tf.keras.Model(eff_model.input, predictions)
    optimizer = Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])   
    # print model summary
    model.summary()

    return model

def plot_hist(history, figure_name):
    """ This function plot the accuracy given history of a model
    Args:
        history (object of dicionary): the history of a model
        figure_name (string): the suffix of the name of figure which willl show the accuracy
    """    
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("logs/" + figure_name + '_accuracy_comparison' + config.options['logs_name'] + '.png')
    # clear plt buffer
    plt.clf()

def train_resnet_model(train_data, train_label, val_data, val_label, tensorboard, pre_trained):
    """This functions trains a simple CNN model based on given parameters
    Args:
        train_data (list): the np array of training data
        train_label (list): the np array of training labels
        val_data (list): the np array of validation data
        val_label (list): the bp array of validation labels
        tensorboard (callback): tensorboard log callback
        pre_trained (bool): this flag defines whether to use pre trained model or not
    """
    print("***Initiating Model : Simple ResNet")
    model = resnet()
    # define checkpoint for saving the best model
    cp_callback = ModelCheckpoint(filepath='checkpoints/resnet_best_' + config.options['logs_name'] + '.h5' ,save_weights_only=False, save_best_only=True, verbose=1)
    # load pre-trained model
    if(pre_trained):
        model.load_weights(config.paths['pre_trained_model_path'])
    # train the model
    history_resnet = model.fit(train_data, train_label, 
                batch_size=config.train_params['batch_size'], 
                epochs=config.train_params['epochs'], 
                validation_data=(val_data, val_label), 
                shuffle=True, 
                callbacks=[tensorboard, cp_callback])
    # save the model of the last epoch
    model.save("models/resnet_last_" + config.options['logs_name'] + ".h5")
    print("***Resnet Model Training Finished***")
    plot_hist(history_resnet, figure_name="resnet")

def train_CNN_model(train_data, train_label, val_data, val_label, tensorboard, pre_trained):
    """This functions trains the resnet model based on given parameters
    Args:
        train_data (list): the np array of training data
        train_label (list): the np array of training labels
        test_data (list): the np array of validation data
        test_label (list): the bp array of validation labels
        tensorboard (callback): tensorboard log callback
        pre_trained (bool): this flag defines whether to use pre trained model or not
    """ 
    print("***Initiating Model : Simple CNN")
    model = simple_CNN_model()
    # define checkpoint for saving the best model
    cp_callback = ModelCheckpoint(filepath='checkpoints/CNN_best_' + config.options['logs_name'] + '.h5' ,save_weights_only=False, save_best_only=True, verbose=1)
    # load pre-trained model
    if(pre_trained):
        model.load_weights(config.paths['pre_trained_model_path'])
    # train the model
    history_CNN = model.fit(train_data, train_label, 
                batch_size=config.train_params['batch_size'], 
                epochs=config.train_params['epochs'], 
                validation_data=(val_data, val_label), 
                shuffle=True, 
                callbacks=[tensorboard, cp_callback])
    # save the model of the last epoch
    model.save("models/CNN_last_" + config.options['logs_name'] + ".h5")
    print("***Simple CNN Model Training Finished***")
    plot_hist(history_CNN, figure_name="CNN")


def train_efficient_net_model(train_data, train_label, val_data, val_label, tensorboard, pre_trained):
    """This functions trains the efficientnet model based on given parameters
    Args:
        train_data (list): the np array of training data
        train_label (list): the np array of training labels
        test_data (list): the np array of validation data
        test_label (list): the bp array of validation labels
        tensorboard (callback): tensorboard log callback
        pre_trained (bool): this flag defines whether to use pre trained model or not
    """    
    print("***Initiating Model : EfficientnetB0")
    model = efficientnet()
    # define checkpoint for saving the best model
    cp_callback = ModelCheckpoint(filepath='checkpoints/efficientnet_best_' + config.options['logs_name'] + '.h5' ,save_weights_only=False, save_best_only=True, verbose=1)

    # load pre trained data
    if(pre_trained):
        model.load_weights(config.paths['pre_trained_model_path'])
    
    # train the model
    history_efficientnet = model.fit(train_data, train_label, 
                batch_size=config.train_params['batch_size'], 
                epochs=config.train_params['epochs'], 
                validation_data=(val_data, val_label), 
                shuffle=True, 
                callbacks=[tensorboard, cp_callback])
    
    # save the model of the last epoch
    model.save("models/efficientnet_last_" + config.options['logs_name'] + ".h5")
    print("***Efficientnet Model Training Finished***")
    plot_hist(history_efficientnet, figure_name="efficientnet")


def train_model(train_mode):
    """This function loads train and validation data and trains the appropiate model based on trainining mode
    Args:
        train_mode (string): training mode one of simple_CNN/resnet/efficienntnet/ensemble
    """  
    # training_mode = config.options['train_mode']
    label_dict = load_label_from_excel(config.paths['label_excel'])
    # generate training data
    train_data, train_label = transform_data('train', config.paths['train_data'], label_dict, config.options['increase_train_image'])
    train_data_len = len(train_data)
    print(f"***Total Train Data {train_data_len}***")
  

    # generate validation data
    val_data, val_label = transform_data('val', config.paths['val_data'], label_dict, config.options['increase_train_image'])
    val_data_len = len(val_data)
    print(f"***Total Validation Data {val_data_len}***")

    # monitor model logs on tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(config.options['logs_name']))

    if train_mode ==  "simple_CNN":
        train_CNN_model(train_data, train_label, val_data, val_label, tensorboard, config.options['pre_trained'])

    elif train_mode == "resnet":
        train_resnet_model(train_data, train_label, val_data, val_label, tensorboard, config.options['pre_trained'])
        
    elif train_mode == "efficientnet":
        train_efficient_net_model(train_data, train_label, val_data, val_label, tensorboard, config.options['pre_trained'])
        
    elif train_mode == "ensemble":
        train_resnet_model(train_data, train_label, val_data, val_label, tensorboard, pre_trained=False)
        train_efficient_net_model(train_data, train_label, val_data, val_label, tensorboard, pre_trained=False)
        print("***Starting the ensemble***")
        # loading the best models from trained resnet and effiecientnet model
        resnet_model = load_model('checkpoints/resnet_best_' + config.options['logs_name'] + '.h5', compile=False)
        efficientnet_model = load_model('checkpoints/efficientnet_best_' + config.options['logs_name'] + '.h5', compile=False)

        # create a list of the models
        models = [resnet_model, efficientnet_model]
        # define the input shape of the model
        model_input = tf.keras.Input(shape=(224, 224, 3))
        # get the output layer of the model
        model_outputs = [model(model_input) for model in models]
        # define average ensemble method on the two modes
        ensemble_output = tf.keras.layers.Average()(model_outputs)
        # define the ensemble model
        ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)  
        # save the ensemble model
        ensemble_model.save('checkpoints/ensemble_model_' + config.options['logs_name'] + '.h5')
        print("finished Ensembling model")


if __name__ == "__main__":
    train_model(config.options['train_mode'])