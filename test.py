import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# https://zhuanlan.zhihu.com/p/343769603
if __name__ == '__main__':
    print('VERSION:', tf.__version__)
    print('GPU:', tf.config.list_physical_devices('GPU'))
