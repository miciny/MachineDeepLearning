from tensorflow import keras


def get_model(shape_one):
    model = keras.models.Sequential()
    # 两个中间层，一个输出层
    model.add(keras.layers.Dense(64,
                                 activation=keras.activations.relu,
                                 input_shape=(shape_one,)))
    model.add(keras.layers.Dense(64,
                                 activation=keras.activations.relu))
    model.add(keras.layers.Dense(1))
    # 选择优化器和损失函数
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
                  loss=keras.losses.mse,
                  metrics=[keras.metrics.mae])
    return model
