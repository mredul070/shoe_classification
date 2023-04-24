import os
import cv2
import numpy as np
import pandas as pd 
import tensorflow as tf

import config

size = 30

def load_label_from_excel(csv_path):
    """This function generates a dictionary mapping their names and labels

    Args:
        excel_path (string): the path where the excel path is kept
    Returns:
        dict: the image and label mapping dictionary
    """    
    # load csv file
    df = pd.read_csv(csv_path)
    # initiate mapping variable
    label_dict = {}
    # populate dictionary
    for _, row in df.iterrows():
        if ".JPG" in row['Name']:
            row['Name'] = row['Name'][:-4]
        label_dict[row['Name']] = row['Label']

    return label_dict



def transform_data(set, path, label_dict, increase_train_image):
    """the functions process the input images for model training
    Args:
        path (string): the path where the images are kept
    Returns:
        np.array, np.array: processed train images and their corresponding labels
    """    
    data = []
    label = []
    for class_name in os.listdir(path):
        for img_name in os.listdir(os.path.join(path, class_name)):
            # for windows
            img_path = path + '/' + class_name + '/' + img_name 

            # read, resize and normalize image
            org_img = cv2.imread(img_path)
            img = cv2.resize(org_img , (224, 224))
            img = tf.keras.utils.normalize(img, axis=1)

            label_name = label_dict[img_name[:-4]]
            img_label = 0 if label_name == 'Adidas' else 1

            # print(img_name, img_label)
            data.append(img)
            label.append(img_label)

            if set == 'train' and increase_train_image:
                # left crop
                crop_left = org_img[:,size:,:]
                crop_left = cv2.resize(crop_left , (224, 224))
                crop_left = tf.keras.utils.normalize(img, axis=1)
                data.append(crop_left)
                label.append(img_label)
                # right crop
                crop_right = org_img[:, :-size, :]
                crop_right = cv2.resize(crop_right , (224, 224))
                crop_right = tf.keras.utils.normalize(img, axis=1)
                data.append(crop_right)
                label.append(img_label)
                # top crop
                crop_top = org_img[size:, :, :]
                crop_top = cv2.resize(crop_top , (224, 224))
                crop_top = tf.keras.utils.normalize(img, axis=1)
                data.append(crop_top)
                label.append(img_label)
                # bottom crop
                crop_bottom = org_img[:-size, :, :]
                crop_bottom = cv2.resize(crop_bottom , (224, 224))
                crop_bottom = tf.keras.utils.normalize(img, axis=1)
                data.append(crop_bottom)
                label.append(img_label)

    print("***Successfully Converted Data***")
        
    return np.array(data), np.array(label, dtype=np.uint8)



if __name__ == "__main__":
    label_dict = load_label_from_excel(config.paths['label_excel'])
    train_data, train_label = transform_dataset('train', config.paths['train_data'], label_dict, config.options['increase_train_image'])
    test_data, test_label = transform_data('test', config.paths['test_data'], label_dict, config.options['increase_train_image'])
    val_data, val_label = transform_data('val', config.paths['val_data'], label_dict, config.options['increase_train_image'])