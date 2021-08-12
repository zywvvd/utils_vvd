import matplotlib.pyplot as plt
import numpy as np
import cv2
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


def image_resize(img_source, shape=None, factor=None, unique_check=False):
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

    if unique_check:
        pixel_list = np.unique(img_source).tolist()
        if len(pixel_list) == 2 and 0 in pixel_list:
            resized_image[resized_image > 0] = np.max(pixel_list)

    return resized_image


def img_normalize(img, mean, std, to_rgb=False):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    img = img.astype('float32')

    mean = np.array(mean, dtype='float32')
    std = np.array(std, dtype='float32')

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def img_rotate(img,
             angle,
             center=None,
             interpolation=cv2.INTER_LINEAR,
             border_mode=cv2.BORDER_CONSTANT,
             border_value=0,
             auto_bound=False,
             ):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    type = img.dtype
    img = img.astype('uint8')
    scale=1.0
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=interpolation,
        borderValue=border_value,
        borderMode=border_mode)
    rotated.astype(type)
    return rotated
