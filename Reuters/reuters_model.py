from tensorflow import keras


def get_model():
    model = keras.models.Sequential()
    # 两个中间层，一个输出层
    model.add(keras.layers.Dense(46,
                                 activation=keras.activations.relu,
                                 input_shape=(10000,)))
    model.add(keras.layers.Dense(46,
                                 activation=keras.activations.relu))
    model.add(keras.layers.Dense(46,
                                 activation=keras.activations.softmax))
    # 选择优化器和损失函数
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
    return model
