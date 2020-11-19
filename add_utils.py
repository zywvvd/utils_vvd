# -*- coding: utf-8 -*-
# @Author: Zhang Yiwei
# @Date:   2020-08-12 17:39:40
# @Last Modified by:   Zhang Yiwei
# @Last Modified time: 2020-08-12 17:39:41
import line_profiler
from functools import wraps


# Line Profiler Decorator
def line_profiling_deco(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        pr = line_profiler.LineProfiler(func)
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        pr.print_stats()
        return result

    return wrapped