import tensorflow as tf
from MNIST.mnist_funcs import show_image, plot_value_array
from tensorflow import keras
import numpy as np


# 保存地址
keras_model_path = "./model/mnist_model"


def _get_model():
    print("==============创建模型==============")
    model = keras.models.Sequential()
    # 两个中间层，一个输出层
    model.add(keras.layers.Dense(512,
                                 activation=keras.activations.relu,
                                 input_shape=(28 * 28,)))
    model.add(keras.layers.Dense(10,
                                 activation=keras.activations.softmax))
    # 选择优化器和损失函数
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
    return model


def _data_and_pre():
    print("==============加载数据==============")
    mn_ist = tf.keras.datasets.mnist

    # 加载数据集
    (train_images, train_labels), (test_images, test_labels) = mn_ist.load_data()

    # print("==============显示图片==============")
    # show_image(train_images[7])
    # show_image(test_images[4])

    print("==============预处理数据==============")
    print(train_images[7].shape)
    train_images = train_images.reshape((60000, 28 * 28))
    print(train_images[7].shape)
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    print(train_images[7].shape)

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


def fit_and_save():
    model = _get_model()
    train_images, train_labels, test_images, test_labels = _data_and_pre()

    print("==============训练中==============")
    history = model.fit(train_images, train_labels, epochs=5, batch_size=128)

    print("==============测试训练集==============")
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_acc:", test_acc)

    print("==============保存模型==============")
    model.save(keras_model_path)  # save() should be called out of strategy scope


def load_and_test():
    train_images, train_labels, test_images, test_labels = _data_and_pre()

    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)

    print("==============测试训练集==============")
    test_loss, test_acc = new_model.evaluate(test_images, test_labels)
    print("new_model test_acc:", test_acc)


def ocr_image():
    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)

    print("==============处理单张图片==============")
    train_images, train_labels, test_images, test_labels = _data_and_pre()
    # train_image = train_images[7]
    train_image = shape_mine_img()

    img = (np.expand_dims(train_image, 0))

    print("==============识别==============")
    predictions_single = new_model.predict(img)
    print(predictions_single[0])
    print(np.argmax(predictions_single[0]))

    plot_value_array(1, predictions_single[0], test_labels)


def shape_mine_img():
    img_path = './data/img.png'
    image = tf.io.read_file(img_path)
    train_images = np.asarray(preprocess_image(image))
    train_images = train_images.reshape((-1, 28 * 28))
    return train_images[0]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, [28, 28])
    return image


if __name__ == "__main__":
    # _data_and_pre()
    # fit_and_save()
    # load_and_test()
    ocr_image()
