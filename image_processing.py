import matplotlib.pyplot as plt
import numpy as np


def show_hist(img):
    rows, cols = img.shape
    hist = img.reshape(rows * cols)
    histogram, bins, patch = plt.hist(hist, 256, facecolor="green", histtype="bar")  # histogram即为统计出的灰度值分布
    plt.xlabel("gray level")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
    return histogram
