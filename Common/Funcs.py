import numpy as np
from Config import characters, char_num


def process_labels():
    # 构建字符索引
    num_letter_dict = dict(enumerate(list(characters)))
    letter_num_dict = dict(zip(num_letter_dict.values(), num_letter_dict.keys()))
    return letter_num_dict


label_dict = process_labels()


# 字符串转成4*62的张量
def text2vec(text):
    text_len = len(text)
    if text_len > char_num:
        raise ValueError(f'验证码最长{char_num}个字符，此字符为{text}，长度为{text_len}')
    vector = np.zeros([char_num, len(characters)])
    for i, c in enumerate(list(text)):
        idx = label_dict[c]
        vector[i][idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    vec = np.argmax(vec, axis=1)
    text = []
    for i, c in enumerate(vec):
        text.append(characters[c])
    return "".join(text)


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
