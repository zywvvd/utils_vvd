import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_hist(img):
    rows, cols = img.shape
    hist = img.reshape(rows * cols)
    histogram, bins, patch = plt.hist(hist, 256, facecolor="green", histtype="bar")  # histogram即为统计出的灰度值分布
    plt.xlabel("gray level")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
    return histogram


def image_resize(img_source, shape=None, factor=None):
    image_H, image_W = img_source.shape[:2]
    if shape is not None:
        return cv2.resize(img_source, shape)

    if factor is not None:
        resized_H = int(round(image_H * factor))
        resized_W = int(round(image_W * factor))
        return cv2.resize(img_source, [resized_W, resized_H])

    else:
        return img_source
