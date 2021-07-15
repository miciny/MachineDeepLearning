from Reuters.reuters_funcs import vec_sequence, draw_vec_loss, draw_vec_accuracy
from Reuters.reuters_model import get_model
import tensorflow as tf


class ReutersLearn:
    def __init__(self):
        self.reuters = tf.keras.datasets.reuters
        self.x_val = list()
        self.partial_x_train = list()
        self.y_val = list()
        self.partial_y_train = list()
        self.x_test = list()
        self.y_test = list()
        self.model = get_model()

    def set_data(self):
        # 加载数据集，num_words=10000意味着您将只保留训练数据中最常见的10,000个单词
        print("==============加载数据中==============")
        (train_data, train_labels), (test_data, test_labels) = self.reuters.load_data(num_words=10000)
        print("==============加载数据完成==============")

        # 列表变成张量,数据进行矢量化
        x_train = vec_sequence(train_data)
        self.x_test = vec_sequence(test_data)
        print("==============数据矢量化完成==============")

        # 标签进行矢量化
        y_train = tf.keras.utils.to_categorical(train_labels)
        self.y_test = tf.keras.utils.to_categorical(test_labels)
        print("==============标签矢量化完成==============")

        # 分离10000个数据，验证集
        self.x_val = x_train[:1000]
        self.partial_x_train = x_train[1000:]
        self.y_val = y_train[:1000]
        self.partial_y_train = y_train[1000:]
        print("==============分离数据完成==============")

    def run(self):
        # 训练
        print("==============训练中==============")
        history = self.model.fit(self.partial_x_train,
                                 self.partial_y_train,
                                 epochs=10,
                                 batch_size=512,
                                 validation_data=(self.x_val, self.y_val))
        print("==============训练完成==============")
        # print(history.history.keys())

        # 打印结果，画图
        print("==============打印图表==============")
        draw_vec_loss(history)
        draw_vec_accuracy(history)

    def test(self):
        print("==============测试数据表现==============")
        results = self.model.evaluate(self.x_test, self.y_test)
        print(results)

        pre = self.model.predict(self.x_test)
        print(pre)


if __name__ == "__main__":
    learn = ReutersLearn()
    learn.set_data()
    learn.run()
    learn.test()
