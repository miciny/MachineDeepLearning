from captcha.image import ImageCaptcha  # pip install captcha
import random
from Config import characters, char_num, img_w, img_h
import os


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=characters, captcha_size=char_num):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image(out_path):
    c_image = ImageCaptcha(width=img_w, height=img_h)
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    c_image.generate(captcha_text)
    c_image.write(captcha_text, out_path + captcha_text + '.png', format='png')  # 写到文件


def gen_captcha_batch(out_path, num):
    for i in range(num):
        print(i)
        gen_captcha_text_and_image(out_path)


if __name__ == '__main__':
    train_path = '../Data/Train'
    vel_path = '../Data/Vel'
    test_path = '../Data/Test'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(vel_path):
        os.mkdir(vel_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    gen_captcha_batch(train_path+'/', 10000)
    gen_captcha_batch(vel_path+'/', 1000)
    gen_captcha_batch(test_path+'/', 1000)
