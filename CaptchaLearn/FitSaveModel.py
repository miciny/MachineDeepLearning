from CaptchaLearn.ModelSet import get_model
from CaptchaLearn.TrainDataSet import get_data, get_test_data
import tensorflow as tf


# 保存地址
keras_model_path = "../CaptchaModel"


def fit_and_save():
    model = get_model()
    train_images, train_labels, vel_images, vel_labels = get_data()

    print("==============训练中==============")
    history = model.fit(train_images, train_labels, epochs=9, batch_size=128)
    print(history.history)

    print("==============测试训练集==============")
    test_loss, test_acc = model.evaluate(vel_images, vel_labels)
    print("test_loss, test_acc:", test_loss, test_acc)

    print("==============保存模型==============")
    model.save(keras_model_path)  # save() should be called out of strategy scope


def load_and_test():
    test_images, test_labels = get_test_data()

    print("==============加载模型==============")
    new_model = tf.keras.models.load_model(keras_model_path)

    print("==============测试训练集==============")
    test_loss, test_acc = new_model.evaluate(test_images, test_labels, verbose=2)
    print("new_model test_loss/test_acc:", test_loss, test_acc)


if __name__ == "__main__":
    fit_and_save()
    load_and_test()
