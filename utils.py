# -*- coding: utf-8 -*-
# @Author: Zhang Yiwei
# @Date:   2020-07-18 02:40:35
# @Last Modified by:   Zhang Yiwei
# @Last Modified time: 2020-08-18 15:51:06
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
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import json
import time
import sys
import cv2 as cv
import logging
import os
import platform
import hashlib
import pickle

from ipdb import set_trace
from functools import wraps


def timer_vvd(func):
    """
    a timer for func
    you could add a @timer_vvd ahead of the fun need to be timed
    Args:
        func (function): a function to be timed

    Outputs:
        time message: a message which tells you how much time the func spent will be printed
    """
    func_name = func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('func: {_funcname_} runing: {_time_}s'.format(_funcname_=func_name, _time_=format(end_time - start_time, '.6f')))
        return res
    return wrapper


def get_current_dir():
    """
    get current dir of the running file
    """
    return os.path.dirname(os.path.realpath(__file__))


def pickle_save(object, save_path):
    """
    将object保存为pickle文件到save_path中
    """
    save_path = save_file_path_check(save_path)
    with open(save_path, 'wb') as fp:
        pickle.dump(object, fp)


def pickle_load(load_path):
    """
    从load_path中读取object
    """
    with open(load_path, 'rb') as fp:
        return pickle.load(fp)


def find_sub_string(string, substring, times):
    """
    find the char position of the substring in string for times-th comes up
    """
    current = 0
    for _ in range(1, times+1):
        current = string.find(substring, current+1)
        if current == -1:
            return -1

    return current


def underline_connection(*str_args, connect_char='_'):
    """
    connect strings in the list with underline
    """
    assert isinstance(str_args, tuple)
    string = ""
    for item in str_args:
        if isinstance(item, list):
            item = underline_connection(*item, connect_char=connect_char)
        if item != '':
            string = string + item + connect_char
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
    """
    返回当前操作系统名称字符串
    """
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


def save_file_path_check(save_file_path):
    """
    检查要保存的文件路径
    - 如果文件已经存在 ： 在文件名与扩展名之间加入当前时间作为后缀 避免覆盖之前的文件并给出提示
    - 如文件不存在 ： 检查文件所在的文件夹目录
    返回检查后的文件路径
    """
    assert isinstance(save_file_path, str)
    if OS_exists(save_file_path):
        main_file_name = get_main_file_name(save_file_path)
        new_base_name = OS_basename(save_file_path).replace(main_file_name, underline_connection(main_file_name, time_stamp()))
        checked_save_file_path = OS_join(OS_dirname(save_file_path), new_base_name)
        print("file path {} already exists, the file will be saved as {} instead.".format(save_file_path, checked_save_file_path))
    else:
        dir_check(OS_dirname(save_file_path))
        assert OS_basename(save_file_path) != ''
        checked_save_file_path = save_file_path
    return checked_save_file_path


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
    """
    获取文件hash值
    """
    assert os.path.exists(file)
    md5_hash = hashlib.md5()
    with open(file, "rb") as fid:
        md5_hash.update(fid.read())
        digest = md5_hash.hexdigest()
    return digest


def my_linux_set_trace(debug=True):
    """
    在Linux 中加入断点
    """
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

    assert np.ndim(in_array) == 3 or np.ndim(in_array) == 2

    if np.ndim(in_array) == 3:
        rows, cols, ndim = in_array.shape
    else:
        rows, cols = in_array.shape

    if (padding_size_3 is None) and (padding_size_4 is None):
        padding_size_1 = max(padding_size_1, 0)
        padding_size_2 = max(padding_size_2, 0)
        assert padding_size_1 >= 0 and padding_size_2 >= 0
        if np.ndim(in_array) == 3:
            padded_array = np.zeros([rows + 2 * padding_size_1, cols + 2 * padding_size_2, ndim], dtype=type(in_array[0][0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_2:cols + padding_size_2, :] = in_array
        elif np.ndim(in_array) == 2:
            padded_array = np.zeros([rows + 2 * padding_size_1, cols + 2 * padding_size_2], dtype=type(in_array[0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_2:cols + padding_size_2] = in_array
        else:
            raise ValueError("np.ndim error")

    else:
        assert (padding_size_3 is not None) and (padding_size_4 is not None), "padding_size_3 padding_size_4 必须都不是none"
        padding_size_1 = max(padding_size_1, 0)
        padding_size_2 = max(padding_size_2, 0)
        padding_size_3 = max(padding_size_3, 0)
        padding_size_4 = max(padding_size_4, 0)
        assert padding_size_1 >= 0 and padding_size_2 >= 0 and padding_size_3 >= 0 and padding_size_4 >= 0
        if np.ndim(in_array) == 3:
            padded_array = np.zeros([rows + padding_size_1 + padding_size_2, cols + padding_size_3 + padding_size_4, ndim], dtype=type(in_array[0][0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_3:cols + padding_size_3, :] = in_array
        elif np.ndim(in_array) == 2:
            padded_array = np.zeros([rows + padding_size_1 + padding_size_2, cols + padding_size_3 + padding_size_4], dtype=type(in_array[0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_3:cols + padding_size_3] = in_array
        else:
            raise ValueError("np.ndim error")

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

    max_value = np.max(image)
    if not is_integer(max_value) and max_value > 1:
        image /= np.max(image)

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
        temp_image = np.zeros([shape[0], shape[1], 3], dtype=type(image[0][0]))
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


def plt_image_show(image, window_name='image show'):
    '''
    更加鲁棒地显示图像包括二维图像,第三维度为1的图像
    '''
    # temp_image = extend_image_channel(image)
    plt.subplot(1, 1, 1)
    plt.imshow(image, cmap='jet')
    plt.title(window_name)
    plt.show()


def draw_RB_map(y_true, y_pred, map_save_path=None):
    assert isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray)
    assert np.ndim(y_pred) == 1
    assert y_pred.shape == y_true.shape

    sorted_ids = np.argsort(y_pred+np.random.rand(y_pred.size)*1e-8)
    sorted_y_true = y_true[sorted_ids]
    ng_rank = np.where(sorted_y_true == 1)[0]
    ok_rank = np.where(sorted_y_true == 0)[0]

    plt.figure(figsize=(25, 3))
    plt.bar(ok_rank, 1, width=1, color='b')
    plt.bar(ng_rank, 1, width=int(ok_rank.size/ng_rank.size/5+1), color='r')
    plt.ylim([0, 1])
    plt.xlim([0, len(ok_rank)+len(ng_rank)])
    if map_save_path is not None:
        plt.savefig(map_save_path)
    plt.show()

    plt.figure(figsize=(25, 3))
    plt.hist(y_pred, bins=255)
    plt.title('ng_score distribution')
    plt.show()


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
    """
    返回当前时间戳字符串
    格式: 年-月-日_时-分-秒
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def vvd_image_preprocess(image):
    """
    vvd 图像预处理
    """
    new_image = image / 127.5 - 1
    return new_image


def json_load(json_path):
    """
    读取json文件并返回内容字典
    """
    try:
        assert OS_exists(json_path)
    except:
        print('file not found !')
    with open(json_path, 'r') as fp:
        return json.load(fp)


def json_save(json_dict, json_path):
    """
    将内容字典保存为json文件
    """
    json_path = save_file_path_check(json_path)
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(json_dict, fp, indent=4, cls=MyEncoder)


def glob_recursively(path, extension):
    """
    在path 路径中递归查找所有扩展名为extension的文件，返回完整路径名列表
    """
    return glob(OS_join(path, '**', '*.' + extension), recursive=True)


def is_integer(num):
    """
    是否是整数，返回bool结果
    """
    return isinstance(num, (int, np.int, np.int32, np.uint8))


def whether_divisible_by(to_be_divided, dividing):
    """
    to_be_divided 是否可以被 dividing 整除，返回bool结果
    """
    assert is_integer(to_be_divided) and is_integer(dividing)
    if to_be_divided % dividing == 0:
        return True
    else:
        return False


if __name__ == '__main__':
    test_name = 'abc/sadf/gsdf.sadf.test'
    strong_printing(test_name)
    print(underline_connection())
