import os
import numpy as np
from OcrImgProgress.PreSetOcrImage import get_dynamic_binary_image
from Config import img_w, img_h
from Common.Funcs import text2vec, vec2text


# 生成数据集
def _gen_train_data(file_path):
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
            # 对图片去噪，
            captcha_image = get_dynamic_binary_image(origin_img_path)
            img_np = np.array(captcha_image)
            x_data.append(img_np)

            # 标签处理成【4， 62】张量
            name = selected_train_file_name.split('.')[0].split('_')[0]
            train_labels = text2vec(name)
            y_data.append(train_labels)
    return x_data, y_data


# 拿到原始数据后，进行处理
def _get_data(path):
    (images, labels) = _gen_train_data(path)
    images = np.array(images).astype(np.float32)
    labels = np.array(labels)
    images = images.reshape((len(images), img_h, img_w, 1))
    images = images.astype('float32') / 255
    # print(images.shape)
    # print(labels.shape)
    # print(labels[0])
    # print(vec2text(labels[0]))
    return images, labels


# 训练和测试集数据
def get_data():
    (train_images, train_labels) = _get_data('../Data/Train')
    (vel_images, vel_labels) = _get_data('../Data/Vel')
    return train_images, train_labels, vel_images, vel_labels


# 测试集数据
def get_test_data():
    (test_images, test_labels) = _get_data('../Data/Test')
    return test_images, test_labels


if __name__ == '__main__':
    get_data()
    get_test_data()
