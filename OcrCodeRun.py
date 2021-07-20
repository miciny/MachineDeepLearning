from Common.PreSetOcrImage import shape_image
import tensorflow as tf
import numpy as np
from Config import img_w, img_h
from RunTimeProperties import pro_dir
from Common.Funcs import vec2text


# 保存地址
keras_model_path = "CaptchaModel"


def ocr_image():
    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)

    print("==============识别==============")
    # real_str = '7KE6'
    # real_str = '6rJH'
    real_str = '0dYM'
    img_path = pro_dir + f'/PredictImages/{real_str}.png'
    out_path = pro_dir + f'/PredictImages/{real_str}_temp.png'

    train_image = shape_image(img_path, img_h=img_h, img_w=img_w, out_path=out_path)
    train_image = (np.expand_dims(train_image, 0))
    predictions_single = new_model.predict(train_image)

    text_list = predictions_single[0]
    text_str = vec2text(text_list)
    print("真实为："+real_str, "识别为："+text_str,
          "\n区分大小写识别结果："+str(text_str == real_str),
          "\n不区分大小写识别结果："+str(text_str.lower() == real_str.lower()))


if __name__ == "__main__":
    ocr_image()
