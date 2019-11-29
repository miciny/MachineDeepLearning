from MNIST.mnist_model import get_model
import tensorflow as tf
from MNIST.mnist_funcs import show_image


class MNISTLearn():
    def __init__(self):
        self.mn_ist = tf.keras.datasets.mnist
        self.model = get_model()

    def set_data(self):
        # 加载数据集
        print("==============加载数据中==============")
        (train_images, train_labels), (test_images, test_labels) = self.mn_ist.load_data()

        print("==============显示图片==============")
        show_image(train_images[4])
        show_image(test_images[4])

        print("==============预处理数据==============")
        train_images = train_images.reshape((60000, 28 * 28))
        train_images = train_images.astype('float32') / 255
        test_images = test_images.reshape((10000, 28 * 28))
        test_images = test_images.astype('float32') / 255

        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)

        print("==============训练中==============")
        history = self.model.fit(train_images,
                                 train_labels,
                                 epochs=5,
                                 batch_size=128)

        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print("test_acc:", test_acc)


if __name__ == "__main__":
    learn = MNISTLearn()
    learn.set_data()
