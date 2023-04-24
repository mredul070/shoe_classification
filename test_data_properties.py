import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# from data_preprocess import generate_train_data

# training image directory
TRAIN_DIR = "dataset/train"
# testing image directory
TEST_DIR = "dataset/test"
# validation image directory
VAL_DIR = "dataset/validation"

def check_img_size(check_dir):
    for class_name in os.listdir(check_dir):
        for img_name in os.listdir(os.path.join(check_dir, class_name)):
            # for general purpose
            # img_path = os.path.join(TRAIN_DIR, img_name)
            # for windows
            img_path = check_dir + '/' + class_name + '/' + img_name 
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            print(img.shape)
            # if img.shape != (224, 224):
                # return "size missmatch"
    return "all are same size"
            



if __name__ == "__main__":
    # train_data = generate_train_data()
    # plt.imshow(train_data[0][0])
    # plt.show()
    print(check_img_size(VAL_DIR))
    # print(check_img_size(TEST_DIR))
    # for class_name in os.listdir(TRAIN_DIR):
    #     for img_name in os.listdir(os.path.join(TRAIN_DIR, class_name)):
    #         # for general purpose
    #         # img_path = os.path.join(TRAIN_DIR, img_name)
    #         # for windows
    #         img_path = TRAIN_DIR + '/' + class_name + '/' + img_name 
    #         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         img = tf.keras.utils.normalize(img, axis=1)
    #         print(img.shape)
    #         # plt.imshow(img)
    #         # plt.show()
    #         break