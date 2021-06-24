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
from numpy.lib.function_base import iterable

from pathlib2 import Path as Path2
from pathlib import Path
import PIL.Image as Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from glob import glob

import numpy as np
import inspect
import json
import time
import sys
import cv2 as cv
import logging
import os
import platform
import hashlib
import pickle
import uuid
import shutil

from tqdm import tqdm

from functools import wraps
from functools import reduce

popular_image_suffixes = ['png', 'jpg', 'jpeg', 'bmp']


def image_formate_transfer(origin_dir, tar_dir, origin_suffix, tar_suffix, recursively=False):

    if origin_suffix.lower() not in popular_image_suffixes:
        raise Warning(f'origin_suffix {origin_suffix} is not an usually image file suffix')

    if tar_suffix.lower() not in popular_image_suffixes:
        raise Warning(f'tar_suffix {tar_suffix} is not an usually image file suffix')

    assert Path(origin_dir).is_dir(), f"origin_dir {origin_dir} does not exist"
    assert Path(tar_dir).is_dir(), f"origin_dir {tar_dir} does not exist"

    image_path_list = glob_recursively(origin_dir, origin_suffix, recursively=recursively)

    for image_path in tqdm(image_path_list, desc=f"converting suffix from {origin_suffix} to {tar_suffix}"):
        img = Image.open(image_path)
        file_name = Path(image_path).stem
        new_file_name = str(Path(tar_dir) / (file_name + '.' + tar_suffix))
        img.save(new_file_name)


def get_mac_address():
    mac=uuid.UUID(int = uuid.getnode()).hex[-12:].upper()
    #return '%s:%s:%s:%s:%s:%s' % (mac[0:2],mac[2:4],mac[4:6],mac[6:8],mac[8:10],mac[10:])
    return ":".join([mac[e:e+2] for e in range(0,11,2)])


def OS_dir_list(dir_path: str):
    """[文件夹下所有文件夹路径]

    Args:
        dir_path (str): [输入文件夹路径]

    Returns:
        [list]: [文件夹下文件夹路径（非递归）]
    """
    dir_name_list = os.listdir(dir_path)
    path_list = list()
    for dir_name in dir_name_list:
        path = OS_join(dir_path, dir_name)
        if OS_isdir(path):
            path_list.append(path)
    return path_list


def crop_data_around_boxes(image, crop_box, cut_box_back=False):
    """make a image crop from a image safely"""

    ndim = image.ndim
    height, width = image.shape[:2]

    crop_box = np.array(crop_box).astype('int32').tolist()

    ori_left, ori_top, ori_right, ori_bottom = 0, 0, width, height

    crop_left, crop_top, crop_right, crop_bottom = crop_box

    assert crop_right > crop_left and crop_bottom > crop_top

    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top

    cut_left = max(crop_left, ori_left)
    cut_right = max(min(ori_right, crop_right), cut_left)
    cut_top = max(ori_top, crop_top)
    cut_bottom = max(min(ori_bottom, crop_bottom), cut_top)

    cut_box = [cut_left, cut_top, cut_right, cut_bottom]

    crop_ori = image[cut_top:cut_bottom, cut_left:cut_right, ...]

    if cut_right - cut_left != crop_width or cut_bottom - cut_top != crop_height:

        # out of boundary
        if ndim == 3:
            crop_ori_temp = np.zeros([crop_height, crop_width, 3], dtype='uint8')
        elif ndim == 2:
            crop_ori_temp = np.zeros([crop_height, crop_width], dtype='uint8')
        else:
            raise RuntimeError(f"error image shape {image.shape} ndim {ndim}")

        win_left = cut_left - crop_left
        win_right = max(cut_right - crop_left, win_left)
        win_top = cut_top - crop_top
        win_bottom = max(cut_bottom - crop_top, win_top)

        crop_ori_temp[win_top:win_bottom, win_left:win_right, ...] = crop_ori
        crop_ori = crop_ori_temp

    if cut_box_back:
        return crop_ori, cut_box
    else:
        return crop_ori


def make_box(center_point, box_x, box_y=None):
    """ build box for a given center-point"""
    box_x = int(box_x)
    if box_y is None:
        box_y = box_x
    else:
        box_y = int(box_y)
    assert box_x > 0 and box_y > 0
    center_x, center_y = center_point

    left = int(round(center_x - box_x // 2))
    right = left + box_x
    top = int(round(center_y - box_y // 2))
    bottom = top + box_y

    box = [left, top, right, bottom]
    return box


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


def file_read_lines(file_path):
    if not OS_exists(file_path):
        print("file {} not found, None will be return".format(file_path))
        return None

    with open(file_path, "r") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            lines[index] = line.strip('\n')
        return lines


def file_write_lines(line_list, file_path, overwrite=False, verbose=False):
    dir_check(OS_dirname(file_path))
    file_path = save_file_path_check(file_path, overwrite, verbose)
    with open(file_path, 'w') as f:
        f.writelines('\n'.join(line_list))


def pickle_save(object, save_path, overwrite=False, verbose=False):
    """
    将object保存为pickle文件到save_path中
    """
    save_path = save_file_path_check(save_path, overwrite, verbose)
    with open(save_path, 'wb') as fp:
        pickle.dump(object, fp)


def pickle_load(load_path):
    """
    从load_path中读取object
    """
    assert isinstance(load_path, str) or is_path_obj(load_path)
    if isinstance(load_path, str):
        load_path = load_path.replace('\\', '/')
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
            string = string + str(item) + connect_char
    string = string[:-1]
    return string


def get_main_file_name(string):
    """
    return file name without extension
    """
    assert isinstance(string, str) or is_path_obj(string)
    if is_path_obj(string):
        string = str(string)
    return os.path.splitext(os.path.basename(string))[0]


def strong_printing(*str_args):
    """
    print string strongly
    """
    assert isinstance(str_args, tuple)
    string = underline_connection(*str_args, connect_char=' ')
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


def save_file_path_check(save_file_path, overwrite=False, verbose=False):
    """
    检查要保存的文件路径
    - 如果文件已经存在 ： 在文件名与扩展名之间加入当前时间作为后缀 避免覆盖之前的文件并给出提示
    - 如文件不存在 ： 检查文件所在的文件夹目录
    返回检查后的文件路径
    """
    if is_path_obj(save_file_path):
        save_file_path = str(save_file_path)

    assert isinstance(save_file_path, str)
    if OS_exists(save_file_path):
        if overwrite:
            checked_save_file_path = save_file_path
            if verbose:
                print("file path {} already exists, the file will be overwrite.".format(save_file_path))
        else:
            main_file_name = get_main_file_name(save_file_path)
            new_base_name = OS_basename(save_file_path).replace(main_file_name, underline_connection(main_file_name, time_stamp()))
            checked_save_file_path = OS_join(OS_dirname(save_file_path), new_base_name)
            if verbose:
                print("file path {} already exists, the file will be saved as {} instead.".format(save_file_path, checked_save_file_path))
    else:
        dir_check(str(Path(save_file_path).parent), verbose)
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


def create_uuid():
    """ create a uuid (universally unique ID) """
    return uuid.uuid1().hex


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

    def vvd_logging(self, *message):
        for message_str in message:
            print(message_str)
            self.logger.info(message_str)

    def vvd_logging_quiet(self, *message):
        for message_str in message:
            self.logger.info(message_str)


def log_init(log_path, quiet=False):
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

    if quiet:
        return Loger_printer(logging).vvd_logging_quiet
    else:
        return Loger_printer(logging).vvd_logging


def dir_exists(dir_path):
    """
    check if dir exists
    """
    dir_path = str(dir_path)
    if not os.path.isdir(dir_path):
        raise TypeError("dir not found")


def uniform_split_char(string, split_char=current_split_char()):
    """
    uniform the split char of a string
    """
    assert isinstance(string, str)
    return string.replace('\\', split_char).replace('/', split_char)


def dir_check(dir_path, verbose=False):
    """
    check if `dir_path` is a real directory path
    if dir not found, make one
    """

    dir_path = str(dir_path)
    assert isinstance(dir_path, str)
    dir_path = uniform_split_char(dir_path)
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
            if verbose:
                print('dirs made: {}'.format(dir_path))
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


def image_read(image_path, channel=-1):
    """
    读取图像，可包含中文路径
    Args:
        image_path ([str]): [图像路径]
        channel (int, optional): [图像通道数，-1为默认，0为灰度]. Defaults to -1.
    """
    image_path = str(image_path)
    return cv.imdecode(np.fromfile(image_path, dtype=np.uint8), channel)


def time_reduce(*data):
    """
    [计算输入数据的乘积]
    """
    data = list(data)
    return reduce(lambda x, y: x*y, data)


def get_function_name():
    '''获取正在运行函数(或方法)名称'''
    # print(sys._getframe().f_code.co_name)
    return inspect.stack()[1][3]


def plt_image_show(*image, window_name='image show', array_res=False, full_screen=True, cmap=None, position=[30, 30], share_xy=False):
    '''
    更加鲁棒地显示图像包括二维图像,第三维度为1的图像
    '''
    image_list = list(image)
    # temp_image = extend_image_channel(image)
    image_num = len(image_list)
    col_num = int(np.ceil(image_num**0.5))
    row_num = int(np.ceil(image_num/col_num))
    if full_screen:
        if current_system() == 'Windows':
            figsize=(18.5, 9.4)
        else:
            figsize=(18.5, 9.4)
    _, ax = plt.subplots(row_num, col_num, figsize=figsize, sharex=share_xy, sharey=share_xy)
    for index, image_item in enumerate(image_list):
        if isinstance(image_item, tuple) or isinstance(image_item, list):
            assert len(image_item) == 2
            image = image_item[0]
            current_name = image_item[1]
            print_name = current_name
        else:
            image = image_item
            print_name = window_name

        if iterable(ax):
            cur_ax = ax[index]
        else:
            cur_ax = ax

        if 'uint8' == image.dtype.__str__():
            cur_ax.imshow(image, cmap=cmap, vmax=np.max(image), vmin=np.min(image))
        elif 'int' in image.dtype.__str__():
            cur_ax(image, cmap=cmap, vmax=np.max(image), vmin=np.min(image))
        else:
            cur_ax(image, cmap=cmap)

        cur_ax.set_title(print_name)

    if not array_res:
        try:
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
        except Exception:
            pass
        plt.show()
    else:
        return convert_plt_to_rgb_image(plt)


def convert_plt_to_rgb_image(plt):
    # Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    rgb_image = image[:, :, :3]
    return rgb_image


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
    plt.plot(data)
    plt.show()


def is_path_obj(path):
    if isinstance(path, Path) or isinstance(path, Path2):
        return True
    else:
        return False


def cv_rgb_imread(image_path):
    """
    按照RGB顺序使用cv读取图像
    """
    image_path = str(image_path)
    image = cv.imread(image_path)
    b, g, r = cv.split(image)
    image = cv.merge([r, g, b])

    return image


def cv_rgb_bgr_convert(image):
    """[convert rgb to bgr or bgr ro rgb]

    Args:
        image ([np.array(uint8)]): [uint8 image]

    Returns:
        [image]: [r and b swapped]
    """
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


def smart_copy(source_file_path, target_path, verbose=False):
    """[复制文件从源到目标，如果目标已经存在则跳过]]

    Args:
        source_file_path ([str]): [源文件路径]
        target_path ([str]): [目标文件夹/目标文件路径]
        verbose (bool, optional): [是否显示信息]. Defaults to False.
    """
    source_file_path = str(source_file_path)
    target_path = str(target_path)
    assert OS_exists(source_file_path)
    if OS_isdir(target_path):
        target_path = OS_join(target_path, OS_basename(source_file_path))
    if OS_exists(target_path):
        if verbose:
            print("{} already exists!".format(target_path))
    else:
        dir_check(Path(target_path).parent)
        shutil.copy(source_file_path, target_path)


def json_load(json_path, verbose=False):
    """
    读取json文件并返回内容字典
    """
    json_path = str(json_path)
    if isinstance(json_path, str):
        json_path = json_path.replace('\\', '/')
    try:
        assert OS_exists(json_path)
    except Exception as e:
        if verbose:
            print('file not found !', e)
    try:
        with open(json_path, 'r') as fp:
            return json.load(fp)
    except Exception as e:
        if verbose:
            print('simple json load failed, try utf-8', e)
    try:
        with open(json_path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception as e:
        if verbose:
            print('utf-8 json load failed, try gbk', e)
    try:
        with open(json_path, 'r', encoding='gbk') as fp:
            return json.load(fp)
    except Exception as e:
        if verbose:
            print('gbk json load failed!', e)        


def json_save(json_dict, json_path, overwrite=False, verbose=False):
    """
    将内容字典保存为json文件
    """
    json_path = str(json_path)
    json_path = save_file_path_check(json_path, overwrite, verbose)
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(json_dict, fp, ensure_ascii=False, sort_keys=False, indent=4, cls=MyEncoder)


def glob_recursively(path, extension, recursively=True):
    """
    在path 路径中递归查找所有扩展名为extension的文件，返回完整路径名列表
    """
    path = str(path)
    if recursively:
        return glob(OS_join(path, '**', '*.' + extension), recursive=True)
    else:
        return glob(OS_join(path, '*.' + extension), recursive=True)


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


def vvd_round(num):
    if iterable(num):
        return np.round(np.array(num)).astype('int32').tolist()
    return int(round(num))


def vvd_ceil(num):
    if iterable(num):
        return np.ceil(np.array(num)).astype('int32').tolist()
    return int(np.ceil(num))


def vvd_floor(num):
    if iterable(num):
        return np.floor(np.array(num)).astype('int32').tolist()
    return int(np.floor(num))


def cv_rgb_imwrite(rgb_image, image_save_path):
    """
    [cv2 save a rgb image]
    Args:
        rgb_image ([np.array]): [rgb image]
        image_save_path ([str/Path]): [image save path]
    """
    bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
    image_save_path = Path(image_save_path)
    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(image_save_path), bgr_image)


def image_show_from_path(file_path):
    """[show image from image file path]

    Args:
        file_path ([str or Path]): [path of image file]
    """
    assert is_path_obj(file_path) or isinstance(file_path, str)
    file_path = str(file_path)
    if not OS_exists(file_path):
        print('file: ', file_path, 'does not exist.')
    else:
        image = cv_rgb_imread(file_path)
        plt_image_show(image)


def erode(mat, iterations=1, kernel_size=3):
    """ dilate 2D binary matrix by one pixel """
    assert isinstance(mat, np.ndarray) and mat.dtype == np.bool
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mat_dilated = cv.erode(mat.astype(np.uint8), kernel, iterations=iterations)
    return mat_dilated > 0


def open_op(mat, iterations=1, kernel_size=3):
    """ dilate 2D binary matrix by one pixel """
    assert isinstance(mat, np.ndarray) and mat.dtype == np.bool
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mat_dilated = cv.erode(mat.astype(np.uint8), kernel, iterations=iterations)
    mat_dilated = cv.dilate(mat_dilated.astype(np.uint8), kernel, iterations=iterations)
    return mat_dilated > 0


def get_gpu_str_as_you_wish(gpu_num_wanted, verbose=0):
    """[get empty gpu index str as needed]

    Args:
        gpu_num_wanted ([int]): [the num of gpu you want]

    Returns:
        [str]: [gpu str returned]
    """
    try:
        import pynvml
    except Exception as e:
        print('can not import pynvml.', e)
        print('please make sure pynvml is installed correctly.')
        print('a simple pip install nvidia-ml-py3 may help.')
        print('now a 0 will be return')
        return '0'

    NUM_EXPAND = 1024 * 1024

    try:
        # 初始化工具
        pynvml.nvmlInit()
    except Exception as e:
        print('pynvml.nvmlInit failed:', e)
        print('now a 0 will be return')
        return '0'

    # 驱动信息
    if verbose:
        print("GPU driver version: ", pynvml.nvmlSystemGetDriverVersion())

    # 获取Nvidia GPU块数
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()
    returned_gpu_num = max(min(gpu_num_wanted, gpuDeviceCount), 0)

    gpu_index_and_free_memory_list = list()
    for index in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_total = info.total / NUM_EXPAND
        gpu_memory_used = info.used / NUM_EXPAND
        gpu_index_and_free_memory_list.append([index, gpu_memory_total - gpu_memory_used])
        if verbose:
            if index == 0:
                device_name = pynvml.nvmlDeviceGetName(handle)
                print('GPU device name:', device_name)
            print(f'gpu {index}: total_memory: {gpu_memory_total} memory_left: {gpu_memory_total - gpu_memory_used}')

    gpu_index_and_free_memory_list.sort(key=lambda x: - x[1])

    gpu_index_picked_list = list()

    for i in range(returned_gpu_num):
        gpu_index_picked_list.append(gpu_index_and_free_memory_list[i][0])

    gpu_index_str = ','.join([str(index) for index in gpu_index_picked_list])
    if verbose:
        print(returned_gpu_num, 'gpu index will be return.')
        print(f'return gpu str: {gpu_index_str}')

    # 关闭工具
    pynvml.nvmlShutdown()

    return gpu_index_str, gpu_index_picked_list


def boxes_painter(rgb_image, box_list, label_list=None, score_list=None, color_list=None):
    """[paint boxex and labels on image]

    Args:
        rgb_image ([np.array(uint8)]): [np array image as type uint8]
        box_list ([list of list of 4 int]): [list of box like [10(xmin), 20(ymin), 50(xmax), 60(ymax)]]
        label_list ([list of int]): [class indexes of boxes in box_list] (could be none)
        class_name_dict ([dict - index: class_name]): [key is index and value is the name in type of str] (could be none)
    Returns:
        [rgb image]: [image with boxes and labels]
    """
    if label_list is not None:
        assert len(label_list) == len(box_list)

    if score_list is not None:
        assert len(score_list) == len(box_list)

    if color_list is not None:
        assert len(color_list) == len(box_list)

    from PIL import ImageFont, ImageDraw, Image
    import matplotlib.font_manager as fm

    color_list_default = [(159, 20, 98), (95, 32, 219), (222, 92, 189), (56, 233, 120), (23, 180, 100), (78, 69, 20), (97, 202, 39), (65, 179, 135), (163, 159, 219)]
    line_thickness = 3

    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    fontsize = 24

    try:
        if current_system() == 'Windows':
            font = ImageFont.truetype('arial.ttf', fontsize)
        else:
            font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)
    except IOError:
        font = ImageFont.load_default()

    text_height = 22

    # draw boxes
    for index, bbox in enumerate(box_list):

        left, top, right, bottom = np.array(bbox).astype('int').tolist()
        if color_list is not None:
            color = color_list[index]
        else:
            if label_list:
                color = color_list_default[label_list[index] % len(color_list)]
            else:
                color = (255, 255, 0)
        # draw box
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=line_thickness, fill=color)

        # draw text
        display_str = ""

        if label_list:
            display_str += str(label_list[index])

        if score_list:
            if display_str != "":
                display_str += ' '
            score = score_list[index]
            display_str += str(format(score, '.3f'))

        text_width, text_height = font.getsize(display_str)

        text_bottom = top

        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left - 1, text_bottom - text_height - 2 * margin), (right + 1, text_bottom)], fill=color)
        if np.mean(np.array(color)) < 250:
            font_color = 'yellow'
        else:
            font_color = 'red'
        draw.text((int(left + (right - left)/2 - text_width/2), text_bottom - text_height - margin), display_str, fill=font_color, font=font)

    # get image with box and index
    array_image_with_box = np.asarray(pil_image)

    return array_image_with_box


if __name__ == '__main__':
    test_name = 'abc/sadf/gsdf.sadf.test'
    strong_printing(test_name)
    print(underline_connection())
    get_gpu_str_as_you_wish(3, verbose=1)
