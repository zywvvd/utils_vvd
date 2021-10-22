from multiprocessing import Pool
from multiprocessing import Manager
import time
import numpy as np
from tqdm import tqdm


def single_process(main_fun, paras, share_list, share_lock, total_size, start_time, catch_exception):

    if catch_exception:
        try:
            res = main_fun(paras)
        except Exception as e:
            print(e)
            share_list.append(None)
            return
    else:
        res = main_fun(paras)

    share_lock.acquire()
    share_list.append(res)
    share_lock.release()

    print(f"{len(share_list)} / {total_size}")
    
    min_time = (time.time() - start_time) / max(1, len(share_list)) * (total_size - max(1, len(share_list))) / 60
    print(f"time left {min_time} min")


def multi_process(main_fun, para_fun, total_size, max_pool_num=8, catch_exception=True):
    start_time = time.time()
    if max_pool_num > 0:
        mp_manager = Manager()
        share_list = mp_manager.list()
        share_lock = mp_manager.Lock()
        pool = Pool(max_pool_num)
        for index in range(total_size):
            paras = para_fun(index)
            pool.apply_async(func=single_process, args=(main_fun, paras, share_list, share_lock, total_size, start_time, catch_exception))
        pool.close()
        pool.join()
        result_list = list(share_list)
    else:
        result_list = list()
        for index in range(total_size):
            res = main_fun(para_fun(index))
            result_list.append(res)
    print(f"mean time {(time.time() - start_time) / len(result_list)}")
    return result_list


if __name__ == '__main__':
    def para_gen(index):
        para_1 = np.random.randint(0, 200)
        para_2 = np.random.randint(0, 200)
        return {'para_1':para_1, 'para_2':para_2}

    def main_fun_v(para_dict):
        para_1 = para_dict['para_1']
        para_2 = para_dict['para_2']
        return para_1 + para_2

    res_list = multi_process(main_fun_v, para_gen, 200)
    pass