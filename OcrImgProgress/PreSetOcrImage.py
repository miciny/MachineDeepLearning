import numpy as np
import cv2
from PIL import Image


# 手动二值化 翻转黑白
def get_static_binary_image(gray_img, out_image_path=None, threshold=140, reverse=True):
    w, h = len(gray_img), len(gray_img[0])
    for y in range(h):
        for x in range(w):
            if reverse:
                if gray_img[x, y] < threshold:
                    gray_img[x, y] = 255
                else:
                    gray_img[x, y] = 0
            else:
                if gray_img[x, y] < threshold:
                    gray_img[x, y] = 0
                else:
                    gray_img[x, y] = 255
    if out_image_path:
        cv2.imwrite(out_image_path, gray_img)
    return gray_img


# 把图片等分cut_num份
def cut_image_to_4(gray_img, cut_num=4):
    cut_image_list = []
    h, w = gray_img.shape
    del_w = int(w / cut_num)
    for index in range(cut_num):
        cut_image = gray_img[0:h, del_w*index:del_w*index+del_w]  # 先用y确定高，再用x确定宽再用x确定宽
        cut_image_list.append(cut_image)
    return cut_image_list


# 重新设置图片大小
def resize_image(file_path, out_img_path, img_h=50, img_w=150):
    image = Image.open(file_path)
    resized_image = image.resize((img_w, img_h), Image.ANTIALIAS)
    resized_image.save(out_img_path)


# 识别之前 去噪 转为向量
def shape_image(img_path, img_h=28, img_w=28, out_path=None):

    origin_img = cv2.imread(img_path)                        # 读取图片
    print(origin_img.shape)
    h, w, t = origin_img.shape
    if h != img_h or w != img_w:
        resize_image(img_path, img_path, img_h, img_w)

    origin_img = get_dynamic_binary_image(img_path)  # 读取图片
    if out_path:
        cv2.imwrite(out_path, origin_img)
    # origin_img = get_static_binary_image(origin_img, out_image_path=img_path)
    origin_img = np.array(origin_img)
    origin_img = origin_img.reshape(img_h, img_w, 1)
    origin_img = origin_img.astype('float32') / 255

    return origin_img


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
