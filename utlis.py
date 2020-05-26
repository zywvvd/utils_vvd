#
# vvd Tool functions 
#

## 整合常用os操作
from os.path import basename as OS_basename
from os.path import join as OS_join
from os.path import exists as OS_exists
from os.path import isdir as OS_isdir

import matplotlib.pyplot as plt
import numpy as np
import json
import time
import cv2 as cv
import math
import logging
import os


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


class Loger_printer():
    """
    日志打印类
    会在控制台与日志同时打印信息    
    """
    def __init__(self,logger):
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
        level = logging.INFO,  

        # 日志格式: 时间、   日志信息
        format = '%(asctime)s: %(message)s',
    
        # 打印日志的时间
        datefmt = '%Y-%m-%d %H:%M:%S',
    
        # 日志文件存放的目录（目录必须存在）及日志文件名
        filename = log_file_path, 
    
        # 打开日志文件的方式
        filemode = open_type
    )    
    
    return Loger_printer(logging)


def dir_exists(dir_path):
    """
    check if dir exists
    """
    if not os.path.isdir(dir_path):
        raise TypeError("dir not found")
    

def dir_check(dir_path):
    """
    check if `dir_path` is a real directory path
    if dir not found, make one
    """
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path) 
        except Exception as err:
            print(f'failed to make dir {dir_path}, error {err}')
        return False
    else:
        return True


def cv_image_show(image,window_name='image show'):
    '''
    show image (for debug)
    press anykey to destory the window 
    
    image: image in numpy 
    window_name: name of the window
    
    image color - bgr
    '''    
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name,image)
    cv.waitKey(0)
    cv.destroyAllWindows()    


def extend_image_channel(image):
    '''
    cv显示三通道图像，本函数将原始图像扩展到三通道
    '''
    shape = image.shape
    
    if len(shape) == 3:
        if shape[2] == 3:
            return image
        elif shape[2] ==1:
            temp_image = np.zeros([shape[0],shape[1],3])
            for i in range(3):
                temp_image[:,:,i] = image[:,:,0]
            return temp_image
        else:
            raise TypeError('image type error')
    elif len(shape) == 2:
        temp_image = np.zeros([shape[0],shape[1],3])
        for i in range(3):
            temp_image[:,:,i] = image
        return temp_image
    else:
        raise TypeError('image type error')
    
    
def image_show(image,window_name='image show'):
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
    ax = fig.add_subplot(1,1,1)
    ax.plot(data)
    fig.show()    
    

def cv_rgb_imread(image_path):
    """
    按照RGB顺序使用cv读取图像
    """
    image = cv.imread(image_path)
    b,g,r = cv.split(image)
    image = cv.merge([r,g,b])
    
    return image

