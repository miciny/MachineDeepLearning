import tensorflow as tf
import numpy as np
import cv2


def cut_image_to_4(gray_img, cut_num=4):
    cut_image_list = []
    h, w = gray_img.shape
    del_w = int(w / cut_num)
    for index in range(cut_num):
        cut_image = gray_img[0:h, del_w*index:del_w*index+del_w]  # 先用y确定高，再用x确定宽再用x确定宽
        cut_image_list.append(cut_image)
    return cut_image_list


def shape_mine_img(img_path='', img_h=28, img_w=28):
    image = tf.io.read_file(img_path)
    train_images = np.asarray(_preprocess_image(image, img_h=img_h, img_w=img_w))
    train_images = train_images.reshape((-1, img_h * img_w))
    return train_images[0]


def _preprocess_image(image, img_h=28, img_w=28):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [img_w, img_h])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, [img_w, img_h])
    return image


def shape_image(img_path, img_h=28, img_w=28):
    x_data = []
    origin_img = get_dynamic_binary_image(img_path)                        # 读取图片
    origin_img = np.array(origin_img)
    x_data.append(origin_img)
    x_data = np.array(x_data).astype(np.float32)
    train_images = x_data.reshape((len(x_data), img_w * img_h))
    return train_images[0]


# 自适应阀值二值化 灰度处理
def get_dynamic_binary_image(origin_img_path):
    origin_img = cv2.imread(origin_img_path)                        # 读取图片
    gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)         # 灰度
    threshold_img = _threshold_img(gray_img)
    blur_img = _interference_gas(threshold_img)
    return blur_img


# 高斯降噪
def _interference_gas(threshold_img):
    kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    blur_img = cv2.filter2D(threshold_img, -1, kernel)               # 高斯降噪

    # 高斯降噪 后 再做一轮二值化
    interference_img = _threshold_img(blur_img, auto=False)
    return interference_img


# 阈值二值化
def _threshold_img(gray_img, auto=True):
    if auto:
        im_res = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 25, 1)  # 自适应阈值
    else:
        ret, im_res = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    return im_res
