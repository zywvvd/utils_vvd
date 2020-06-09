import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.metrics import binary_crossentropy
from keras.metrics import  categorical_crossentropy



def multi_category_focal_loss(y_true, y_pred, gamma = 2., alpha = 0.75):
    
    epsilon = 1.e-7
    gamma=gamma
    alpha = tf.constant(alpha, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl)
    return loss


if __name__ =='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    Y_true = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1]])
    Y_pred = np.array([[0.9, 0.99, 0.8, 0.97, 0.85, 0.9, 0.95, 0.91, 0.99, 1]], dtype=np.float32)
    
    print(K.eval(multi_category_focal_loss(Y_true, Y_pred)))
    
    epsilon = 1.e-7
    
    y_true = tf.cast(Y_true, tf.float32)
    y_pred = tf.clip_by_value(Y_pred, epsilon, 1. - epsilon)
    
    #print(K.eval(categorical_crossentropy(y_true, y_pred)))
    
    y_true_0 = tf.cast(Y_true[0], tf.float32)
    y_pred_0 = tf.clip_by_value(Y_pred[0], epsilon, 1. - epsilon)
    
    
    print(K.eval(binary_crossentropy(y_true_0, y_pred_0))/2)
    
    
