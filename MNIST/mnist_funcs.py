import matplotlib.pyplot as plt


# 显示数据
def show_image(digit):
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()


