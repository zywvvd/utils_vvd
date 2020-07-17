#
# vvd Tool functions
#

# 整合常用os操作
from os.path import basename as OS_basename
from os.path import join as OS_join
from os.path import exists as OS_exists
from os.path import isdir as OS_isdir
from os.path import dirname as OS_dirname

import matplotlib.pyplot as plt
import numpy as np
import json
import time
import sys
import cv2 as cv
import math
import logging
import os
import platform
import hashlib

from ipdb import set_trace


def underline_connection(*str_args):
    """
    connect strings in the list with underline
    """
    assert isinstance(str_args, tuple)
    string = ""
    for item in str_args:
        string = string + item + '_'
    string = string[:-1]
    return string


def get_main_file_name(string):
    """
    return file name without extension
    """
    assert isinstance(string, str)
    return os.path.splitext(os.path.basename(string))[0]


def strong_printing(string):
    """
    print string strongly
    """
    print()
    boudary_size = int(max(40, len(string)*1.4))
    split_string = boudary_size*'#'
    print(split_string)
    space_size = (boudary_size - len(string))//2
    print(space_size*' '+string.upper())
    print(split_string)
    print()


def current_system():
    return platform.system()


def current_split_char():
    """
    返回当前操作系统的路径分隔符
    """
    if current_system() == 'Windows':
        return '\\'
    elif current_system() == 'Linux':
        return '/'
    else:
        return '/'


def encode_chinese_to_unicode(input_string):
    '''
    将中文转换为 unicode #Uxxxx 形式
    '''
    unicode_string = ''
    for char in input_string:
        if ord(char) > 255:
            char = "%%U%04x" % ord(char)
        unicode_string += char
    unicode_string = unicode_string.replace('%', '#')
    return unicode_string


def get_file_hash_code(file):
    assert os.path.exists(file)
    md5_hash = hashlib.md5()
    with open(file, "rb") as fid:
        md5_hash.update(fid.read())
        digest = md5_hash.hexdigest()
    return digest


def my_linux_set_trace(debug=False):
    if debug:
        Current_System = current_system()
        if Current_System == 'Linux':
            set_trace()


def zero_padding(in_array, padding_size_1, padding_size_2, padding_size_3=None, padding_size_4=None):
    """
    四周补零，以此避免边界判断(仅用于三通道图像)

    输入：
    :in_array: 输入矩阵 np.array (rows, cols, 3)

    (padding_size_3-4 为 None 时)
    :padding_size_1:  上下补零行数
    :padding_size_2:  左右补零列数

    (padding_size_3-4 均不为 None 时)
    :padding_size_1:  上补零行数
    :padding_size_2:  下补零行数
    :padding_size_3:  左补零列数
    :padding_size_4:  右补零列数

    输出：
    :padded_array: 补零后的图像（新建矩阵，不修改原始输入）
    """

    assert np.ndim(in_array) == 3

    rows, cols, ndim = in_array.shape

    if (padding_size_3 is None) and (padding_size_4 is None):
        assert padding_size_1 >= 0 and padding_size_2 >= 0

        padded_array = np.zeros(
            [rows + 2 * padding_size_1, cols + 2 * padding_size_2, ndim], dtype=type(in_array[0][0][0]))
        padded_array[padding_size_1:rows + padding_size_1,
                     padding_size_2:cols + padding_size_2, :] = in_array

    else:
        assert (not padding_size_3 is None) and (
            not padding_size_4 is None), "padding_size_3 padding_size_4 必须都不是none"
        assert padding_size_1 >= 0 and padding_size_2 >= 0 and padding_size_3 >= 0 and padding_size_4 >= 0

        padded_array = np.zeros([rows + padding_size_1 + padding_size_2, cols +
                                 padding_size_3 + padding_size_4, ndim], dtype=type(in_array[0][0][0]))
        padded_array[padding_size_1:rows + padding_size_1,
                     padding_size_3:cols + padding_size_3, :] = in_array

    return padded_array


class MyEncoder(json.JSONEncoder):
    """
    自定义序列化方法，解决 TypeError - Object of type xxx is not JSON serializable 错误
    使用方法：
    在json.dump时加入到cls中即可，例如：
    json.dumps(data, cls=MyEncoder) 
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


class Sys_Logger(object):
    '''
    修改系统输出流
    '''

    def __init__(self, fileN="Default.log"):

        self.terminal = sys.stdout
        if OS_exists(fileN):
            self.log = open(fileN, "a")
        else:
            self.log = open(fileN, "w")

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


class Loger_printer():
    """
    日志打印类
    会在控制台与日志同时打印信息    
    """

    def __init__(self, logger):
        self.logger = logger

    def vvd_logging(self, message):
        self.logger.info(message)
        print(message)


def log_init(log_path):
    """
    initialize logging 
    save the logging object in `config.Parameters.Logging_Object`

    after this operation,
    we could save logs with simple orders such as `logging.debug('test debug')` `logging.info('test info')` 
    logging level : debug < info < warning <error < critical

    Loger_printer.vvd_logging('test')
    """
    dir_name = os.path.dirname(log_path)

    dir_check(dir_name)

    log_file_path = log_path

    if os.path.exists(log_file_path):
        # open log file as  mode of append
        open_type = 'a'
    else:
        # open log file as  mode of write
        open_type = 'w'

    logging.basicConfig(

        # 日志级别,logging.DEBUG,logging.ERROR
        level=logging.INFO,

        # 日志格式: 时间、   日志信息
        format='%(asctime)s: %(message)s',

        # 打印日志的时间
        datefmt='%Y-%m-%d %H:%M:%S',

        # 日志文件存放的目录（目录必须存在）及日志文件名
        filename=log_file_path,

        # 打开日志文件的方式
        filemode=open_type
    )

    logging.StreamHandler()

    return Loger_printer(logging)


def dir_exists(dir_path):
    """
    check if dir exists
    """
    if not os.path.isdir(dir_path):
        raise TypeError("dir not found")


def uniform_split_char(string, split_char=current_split_char()):
    """
    uniform the split char of a string
    """
    assert isinstance(string, str)
    return string.replace('\\', split_char).replace('/', split_char)


def dir_check(dir_path):
    """
    check if `dir_path` is a real directory path
    if dir not found, make one
    """
    assert isinstance(dir_path, str)
    dir_path = uniform_split_char(dir_path)
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as err:
            print(f'failed to make dir {dir_path}, error {err}')
        return False
    else:
        return True


def cv_image_show(image, window_name='image show'):
    '''
    show image (for debug)
    press anykey to destory the window 

    image: image in numpy 
    window_name: name of the window

    image color - bgr
    '''
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extend_image_channel(input_image):
    '''
    cv显示三通道图像，本函数将原始图像扩展到三通道
    '''
    image = input_image.copy()

    shape = image.shape
    if 0 < np.max(image) <= 1:
        image = (255*image).astype('uint8')

    if len(shape) == 3:
        if shape[2] == 3:
            return image
        elif shape[2] == 1:
            temp_image = np.zeros([shape[0], shape[1], 3])
            for i in range(3):
                temp_image[:, :, i] = image[:, :, 0]
            return temp_image
        else:
            raise TypeError('image type error')
    elif len(shape) == 2:
        temp_image = np.zeros([shape[0], shape[1], 3])
        for i in range(3):
            temp_image[:, :, i] = image
        return temp_image
    else:
        raise TypeError('image type error')


def image_show(image, window_name='image show'):
    '''
    更加鲁棒地显示图像包括二维图像,第三维度为1的图像
    '''
    temp_image = extend_image_channel(image)
    cv_image_show(image=temp_image, window_name=window_name)


def data_show(data):
    '''
    show data in a chart
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data)
    fig.show()


def cv_rgb_imread(image_path):
    """
    按照RGB顺序使用cv读取图像
    """
    image = cv.imread(image_path)
    b, g, r = cv.split(image)
    image = cv.merge([r, g, b])

    return image


def time_stamp():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def vvd_image_preprocess(image):
    new_image = image / 127.5 - 1
    return new_image


if __name__ == '__main__':
    test_name = 'abc/sadf/gsdf.sadf.test'
    strong_printing(test_name)
    print(underline_connection())

