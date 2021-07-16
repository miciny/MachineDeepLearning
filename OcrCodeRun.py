from OcrImgProgress.PreSetOcrImage import shape_mine_img, get_dynamic_binary_image, cut_image_to_4
from Utils.ShowPlot import plot_value_array, show_image
import tensorflow as tf
import numpy as np
from Config import characters, img_w, img_h, char_num
import cv2


# 保存地址
keras_model_path = "./ChartModel"


def ocr_image():
    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)

    print("==============识别==============")
    img_path = './ImageToOcr/0Exk_captcha742.png'
    captcha_image = get_dynamic_binary_image(img_path)
    # 切割图片 实际上识别单字母
    captcha_image_cut_list = cut_image_to_4(captcha_image)
    for index, captcha_image_cut in enumerate(captcha_image_cut_list):
        out_img_path = './ImageToOcr/{}.jpg'.format(index)
        cv2.imwrite(out_img_path, captcha_image_cut)

        train_image = shape_mine_img(out_img_path, img_h=img_h, img_w=int(img_w / char_num))
        img = (np.expand_dims(train_image, 0))

        predictions_single = new_model.predict(img)
        index = np.argmax(predictions_single[0])
        print(index, characters[index])


def ocr_single():
    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)
    out_img_path = './ImageToOcr/2.jpg'
    train_image = shape_mine_img(out_img_path, img_h=img_h, img_w=int(img_w / char_num))
    img = (np.expand_dims(train_image, 0))

    predictions_single = new_model.predict(img)
    index = np.argmax(predictions_single[0])
    print(index, characters[index])
    plot_value_array(predictions_single[0], list(characters))


if __name__ == "__main__":
    # ocr_image()
    ocr_single()
