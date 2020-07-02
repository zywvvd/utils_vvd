import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.metrics import binary_crossentropy
from keras.metrics import categorical_crossentropy


def iou_keras(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(y_pred, label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou_keras(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())

    mean_iou = K.variable(0)

    thre_list = list(np.arange(0.0000001, 0.99, 0.05))

    for thre in thre_list:

        y_pred_temp = K.cast(y_pred >= thre, K.floatx())
        y_pred_temp = K.cast(K.equal(y_pred_temp, label), K.floatx())
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred_temp)
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true) + K.sum(y_pred_temp) - intersection
        iou = K.switch(K.equal(union, 0), 1.0, intersection / union)
        mean_iou = mean_iou + iou
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return mean_iou / len(thre_list)


def iou_numpy(y_true, y_pred):

    intersection = np.sum(np.multiply(y_true.astype('bool'), y_pred == 1))
    union = np.sum((y_true.astype('bool')+y_pred.astype('bool')) > 0)

    return intersection/union


def mean_iou_numpy(y_true, y_pred):

    iou_list = []
    for thre in list(np.arange(0.0000001, 0.99, 0.05)):
        y_pred_temp = y_pred >= thre
        iou = iou_numpy(y_true, y_pred_temp)
        iou_list.append(iou)

    # print(iou_list)
    return np.mean(iou_list)


def multi_category_focal_loss(y_true, y_pred, gamma=2., alpha=0.25):

    epsilon = 1.e-7
    gamma = gamma
    alpha = tf.constant(alpha, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = (-1) * tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl)
    return loss


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    Y_true = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1]])
    Y_pred = np.array(
        [[0.9, 0.99, 0.8, 0.97, 0.85, 0.9, 0.95, 0.91, 0.99, 1]], dtype=np.float32)

    print(K.eval(multi_category_focal_loss(Y_true, Y_pred)))

    epsilon = 1.e-7

    y_true = tf.cast(Y_true, tf.float32)
    y_pred = tf.clip_by_value(Y_pred, epsilon, 1. - epsilon)

    # print(K.eval(categorical_crossentropy(y_true, y_pred)))

    y_true_0 = tf.cast(Y_true[0], tf.float32)
    y_pred_0 = tf.clip_by_value(Y_pred[0], epsilon, 1. - epsilon)

    print(K.eval(binary_crossentropy(y_true_0, y_pred_0))/2)
