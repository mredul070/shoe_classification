import os
import cv2
import numpy as np
import tensorflow as tf

import config
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def process_img(img_path):
    """Process the input for inference
    Args:
        img_path (string): the imaged path from where input image is kept
    Returns:
        np.array: the processed input image
    """    
    img = cv2.imread(img_path)
    img = cv2.resize(img , (224, 224))
    img = img.reshape((1, 224, 224, 3))
    img = tf.keras.utils.normalize(img, axis=1)
    return np.array(img)

def get_prediction(img_path, model_path):
    """this function get the prediction of a given image from a given model
    Args:
        img_path (str): the image path from where input image is kept
        model_path (_type_): the model path from where required model is kept
    Returns:
        string: predicted class name of the input image
    """    
    # img_path = ROOT_IMG_PATH + "/" + class_name + "/" + img_name
    processed_img = process_img(img_path)
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(processed_img)
    label_number = np.argmax(prediction)
    class_name = 'Adidas' if label_number == 0 else 'Nike'
    return class_name

if __name__ == "__main__":
    output_class_name = get_prediction(config.paths['inference_img_path'], config.paths['inference_model_path'])
    img_name = config.paths['inference_img_path'].split("/")[-1]
    print(f"The predicted label for image {img_name} is {output_class_name}")