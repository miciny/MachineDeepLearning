import matplotlib.pyplot as plt
import numpy as np


# 显示数据
def show_image(digit):
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()


def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(len(true_label)))
    plt.yticks([])
    this_plot = plt.bar([i for i in true_label], predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    this_plot[predicted_label].set_color('red')
