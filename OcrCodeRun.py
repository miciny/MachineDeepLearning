from OcrImgProgress.PreSetOcrImage import shape_mine_img, get_dynamic_binary_image, cut_image_to_4, \
    shape_image, resize_image, get_static_binary_image
from Utils.ShowPlot import plot_value_array, show_image
import tensorflow as tf
import numpy as np
from Config import characters, img_w, img_h, cut_num
import cv2
from RunTimeProperties import pro_dir
import time


# 保存地址
keras_model_path = "./ChartModel"


def ocr_image():
    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)

    print("==============识别==============")
    img_path = pro_dir + '/ImageToOcr/0Exk_captcha742.png'

    # train_image = shape_image(img_path, img_h=img_h, img_w=int(img_w / cut_num))
    # img = (np.expand_dims(train_image, 0))
    # predictions_single = new_model.predict(img)
    # print(len(predictions_single))
    # print(predictions_single)
    # index = np.argmax(predictions_single[0])
    # print(index, characters[index])

    resize_image(img_path, img_path)      # 调整大小
    captcha_image = get_dynamic_binary_image(img_path)
    # 切割图片 实际上识别单字母
    captcha_image_cut_list = cut_image_to_4(captcha_image)

    out_str = ''
    for index, captcha_image_cut in enumerate(captcha_image_cut_list):
        out_img_path = pro_dir + '/ImageToOcr/{}.jpg'.format(index)
        cv2.imwrite(out_img_path, captcha_image_cut)
        train_image = shape_image(out_img_path, img_h=img_h, img_w=int(img_w / cut_num))
        img = (np.expand_dims(train_image, 0))

        predictions_single = new_model.predict(img)
        index = np.argmax(predictions_single[0])
        print(index, characters[index])
        out_str += characters[index]
    print('0Exk', out_str, out_str == '0Exk')


def ocr_single():
    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)
    out_img_path = './ImageToOcr/3.jpg'
    train_image = shape_image(out_img_path, img_h=img_h, img_w=int(img_w / cut_num))
    img = (np.expand_dims(train_image, 0))

    predictions_single = new_model.predict(img)
    index = np.argmax(predictions_single[0])
    print(index, characters[index])
    plot_value_array(predictions_single[0], list(characters))


def oct_other_img():
    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)

    print("==============识别==============")
    img_path = pro_dir + '/ImageToOcr/img.png'

    captcha_image = get_dynamic_binary_image(img_path)

    # captcha_image = get_static_binary_image(captcha_image, img_path)
    contours, hierarchy = cv2.findContours(captcha_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('轮廓数量: ', len(contours))

    size_list = []
    size_list_temp = dict()
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        size_list.append(w * h)
        size_list_temp[str(w*h)] = (x, y, w, h)

    size_list.sort(reverse=True)

    for index, size in enumerate(size_list[:4]):
        x, y, w, h = size_list_temp[str(size)]
        # 把矩形框里的单图存
        off_set = 5
        cut_image = captcha_image[y - off_set:y + h + off_set, x - off_set:x + w + off_set]  # 先用y确定高，再用x确定宽再用x确定宽
        cut_image_path = pro_dir + '/ImageToOcr/{}.png'.format(index)
        cv2.imwrite(cut_image_path, cut_image)

        resize_image(cut_image_path, cut_image_path, img_w=int(img_w / cut_num), img_h=img_h)  # 调整大小
        train_image = shape_image(cut_image_path, img_h=img_h, img_w=int(img_w / cut_num))
        img = (np.expand_dims(train_image, 0))

        predictions_single = new_model.predict(img)
        index = np.argmax(predictions_single[0])
        print(index, characters[index])


if __name__ == "__main__":
    ocr_image()
    # ocr_single()
    # oct_other_img()
