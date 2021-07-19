from tensorflow import keras
from Config import characters, img_w, img_h, cut_num


def get_model():
    print("==============创建模型==============")
    model = keras.models.Sequential()
    # 两个中间层，一个输出层
    model.add(keras.layers.Dense(512,
                                 activation=keras.activations.relu,
                                 input_shape=(img_h * int(img_w / cut_num),)))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(characters),
                                 activation=keras.activations.softmax))
    # 选择优化器和损失函数
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
    return model
