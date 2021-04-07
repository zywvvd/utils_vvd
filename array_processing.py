import numpy as np


def get_Several_MinMax_Array(np_arr, several):
    """
    获取numpy数值中最大或最小的几个数
    :param np_arr:  numpy数组
    :param several: 最大或最小的个数（负数代表求最大，正数代表求最小）
    :return:
        several_min_or_max: 结果数组
    """
    np_arr = np.array(np_arr)
    if several > 0:
        index_pos = np.argpartition(np_arr, several)[:several]
    else:
        index_pos = np.argpartition(np_arr, several)[several:]
    several_min_or_max = np_arr[index_pos]
    return index_pos, several_min_or_max
