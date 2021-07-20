from tensorflow import keras
from Config import characters, char_num
import tensorflow as tf


def get_model():
    print("==============创建模型==============")
    model = keras.models.Sequential()

    # 卷积核池化操作尽可能的减少计算量
    # 图片像素不高，所以使用的卷积核和池大小不能太大，优先考虑3 * 3 和5 * 5 的卷积核，池大小使用2 * 2
    # 按照下面的神经网络模型，卷积池化以后的输出应该是128 * 17 * 5 = 10880
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(64, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    # 输出值看成4组，需要将输出值调整为(4, 62)的数组
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(char_num * len(characters)))
    model.add(tf.keras.layers.Reshape([char_num, len(characters)]))

    # 多分类问题
    model.add(tf.keras.layers.Softmax())

    # tensorflow2.0中softmax对应的损失函数是categorical_crossentropy
    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model
