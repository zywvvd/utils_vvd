import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.core.fromnumeric import resize
from numpy.lib.function_base import iterable


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
        resized_image = cv2.resize(img_source, shape)

    elif factor is not None:
        if iterable(factor):
            assert len(factor) == 2
            factor_x, factor_y = factor
        else:
            factor_x, factor_y = factor, factor

        resized_H = int(round(image_H * factor_y))
        resized_W = int(round(image_W * factor_x))

        resized_image = cv2.resize(img_source, [resized_W, resized_H])

    elif shape is None and factor is None:
        resized_image = img_source
    else:
        raise RuntimeError

    pixel_list = np.unique(img_source).tolist()
    if len(pixel_list) == 2 and 0 in pixel_list:
        resized_image[resized_image > 0] = np.max(pixel_list)

    return resized_image
