import os
import numpy as np
from OcrImgProgress.PreSetOcrImage import get_dynamic_binary_image, cut_image_to_4
from Config import characters, img_w, img_h, cut_num
import tensorflow as tf
import cv2


def _process_labels(labels):
    """将标签字符转换成数字张量"""
    # 构建字符索引
    num_letter_dict = dict(enumerate(list(characters)))
    letter_num_dict = dict(zip(num_letter_dict.values(), num_letter_dict.keys()))

    ret = []
    for label in labels:
        arr = [letter_num_dict[i] for i in label]
        ret.append(arr)

    return np.array(ret)


def _gen_train_data(file_path):
    """
    生成数据集
    :param file_path: 存filePath文件夹获取全部图片处理
    :return: x_data:图片数据，shape=(num, 20, 80),y_data:标签信息, shape=(num, 4)
    """

    # 返回指定的文件夹包含的文件或文件夹的名字的列表。
    train_file_name_list = os.listdir(file_path)
    # 返回值
    x_data = []
    y_data = []

    # 对每个图片单独处理
    for index, selected_train_file_name in enumerate(train_file_name_list):
        if selected_train_file_name.endswith('.png'):
            # 获取图片对象
            origin_img_path = os.path.join(file_path, selected_train_file_name)
            # 对图片去噪，后面对这个方法单独说明
            captcha_image = get_dynamic_binary_image(origin_img_path)

            # 切割图片 实际上识别单字母
            captcha_image_cut_list = cut_image_to_4(captcha_image)
            for i, captcha_image_cut in enumerate(captcha_image_cut_list):
                img_np = np.array(captcha_image_cut)
                x_data.append(img_np)

            y_temp = np.array(list(selected_train_file_name.split('_')[0]))
            for y in y_temp:
                y_data.append(y)

    x_data = np.array(x_data).astype(np.float32)
    y_data = np.array(y_data)
    return x_data, y_data


def get_data():
    # 生成训练集--train_data_dir(训练文件验证码路径)
    (train_images, train_labels) = _gen_train_data('../Data/Train')
    # 生成测试集--test_data_dir(测试文件验证码路径)
    (vel_images, vel_labels) = _gen_train_data('../Data/Vel')
    # print(train_images.shape)
    # print(vel_images.shape)

    train_images = train_images.reshape((len(train_images), int(img_w / cut_num) * img_h))
    train_images = train_images.astype('float32') / 255
    vel_images = vel_images.reshape((len(vel_images), int(img_w / cut_num) * img_h))
    vel_images = vel_images.astype('float32') / 255

    train_labels = _process_labels(train_labels)
    vel_labels = _process_labels(vel_labels)
    train_labels = tf.keras.utils.to_categorical(train_labels)
    vel_labels = tf.keras.utils.to_categorical(vel_labels)

    return train_images, train_labels, vel_images, vel_labels


def get_test_data():
    # 生成测试集--test_data_dir(测试文件验证码路径)
    (test_images, test_labels) = _gen_train_data('../Data/Test')
    # print(train_images.shape)
    # print(test_images.shape)

    test_images = test_images.reshape((len(test_images), int(img_w / cut_num) * img_h))
    test_images = test_images.astype('float32') / 255

    test_labels = _process_labels(test_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    return test_images, test_labels


if __name__ == '__main__':
    get_data()
