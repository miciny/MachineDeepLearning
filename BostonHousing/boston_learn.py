from BostonHousing.boston_funcs import draw_vec_mae
import numpy as np
from BostonHousing.boston_model import get_model
import tensorflow as tf


class BostonHosingLearn():
    def __init__(self):
        self.boston_housing = tf.keras.datasets.boston_housing
        self.model = None

    def set_data(self):
        # 加载数据集
        print("==============加载数据中==============")
        (train_data, train_targets), (test_data, test_targets) = self.boston_housing.load_data()

        mean = train_data.mean(axis=0)
        train_data -= mean
        std = train_data.std(axis=0)
        train_data /= std
        test_data -= mean
        test_data /= std
        print("==============加载数据完成==============")

        #  K-fold 验证集
        k = 4
        num_val_samples = len(train_data) // 4
        num_epochs = 200
        all_mae = list()
        for i in range(k):
            val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
            val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
            partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                                train_data[(i + 1) * num_val_samples:]],
                                                axis=0)
            partial_train_target = np.concatenate([train_targets[:i * num_val_samples],
                                                  train_targets[(i + 1) * num_val_samples:]],
                                                  axis=0)

            self.model = get_model(train_data.shape[1])

            print('==============训练中', i, "/", k, "==============")
            history = self.model.fit(partial_train_data,
                                     partial_train_target,
                                     validation_data=(val_data, val_targets),
                                     epochs=num_epochs,
                                     batch_size=1,
                                     verbose=0)

            mae_history = history.history['val_mean_absolute_error']
            all_mae.append(mae_history)

        print(all_mae)
        average_mae_history = [np.mean([x[i] for x in all_mae]) for i in range(num_epochs)]
        draw_vec_mae(average_mae_history)


if __name__ == "__main__":
    learn = BostonHosingLearn()
    learn.set_data()
