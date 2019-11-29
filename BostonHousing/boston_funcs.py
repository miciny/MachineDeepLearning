import matplotlib.pyplot as plt


# 绘制图表
def draw_vec_mae(average_mae_history):
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
           previous = smoothed_points[-1]
           smoothed_points.append(previous * factor + point * (1 - factor))
        else:
           smoothed_points.append(point)
    return smoothed_points
