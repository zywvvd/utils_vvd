#
# image calssification 网络回调函数
#

import os
import math

from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau


class ModelCheckpointAfter(ModelCheckpoint):
    """
    每个epoch结束保存模型
    """

    def __init__(self, epoch, filepath, monitor='acc', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):

        super().__init__(filepath, monitor, verbose,
                         save_best_only, save_weights_only, mode, period)
        self.after_epoch = epoch
        self.index = 1

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.after_epoch:
            super().on_epoch_end(self.index, logs)
            self.index += 1

    def on_batch_end(self, batch, logs=None):
        # print(f'batch: {batch}')
        # print(f'logs : {logs}')
        pass


def my_ReduceLROnPlateau(decay_factor, patience, min_lr):

    lr_reducer = ReduceLROnPlateau(
        monitor='loss', factor=decay_factor, cooldown=0, patience=patience, min_lr=min_lr)

    return lr_reducer


class ParallelModelCheckpoint(ModelCheckpointAfter):
    '''
    多gpu checkpoint
    '''

    def __init__(self, model, epoch, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(epoch, filepath,
                                                      monitor, verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def model_checkpoint_after(epoch, steps, path, monitor, save_best_only, ParallelModel):
    """
    每个epoch结束保存模型
    """
    if not (os.path.exists(path)):
        os.mkdir(path)
    pattern = os.path.join(path, 'epoch-{epoch:03d}-{' + monitor + ':.4f}.h5')

    if ParallelModel:
        return ParallelModelCheckpoint(ParallelModel, epoch, steps, filepath=pattern, monitor=monitor, save_best_only=save_best_only)
    else:
        return ModelCheckpointAfter(epoch, steps, filepath=pattern, monitor=monitor,
                                    save_best_only=save_best_only, mode='auto')


class Data_Shuffle(Callback):
    """
    每个epoch结束随机化训练数据
    """

    def __init__(self, generator):
        self.generator = generator

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.generator.on_epoch_end()

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


def learning_rate_step_decay(step_size, decay, verbose=1):
    """
    学习率衰减，setp 衰减
    """
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr
        return lr

    return LearningRateScheduler(schedule, verbose=verbose)


def learning_rate_cosine_decay_with_warmup_and_cycle(
        learning_rate_base=0.00004,
        cosine_total_step=10,
        warmup_learning_rate=4e-6,
        warmup_steps=3,
        least_learnning_rate=3e-7,
        cycle=True,
        t_mul=1.7,
        m_mul=0.6,
        verbose=1):
    """
    每批次带有warmup余弦退火学习率计算
    :param global_step: 当前到达的步数
    :param learning_rate_base: warmup之后的基础学习率
    :param total_steps: 总需要批次数
    :param warmup_learning_rate: warmup开始的学习率
    :param warmup_steps:warmup学习率 步数
    :param hold_base_rate_steps: 预留总步数和warmup步数间隔
    :return:
    """

    def cosine_decay_with_warmup_schedule(epoch, lr=None):

        def get_cycle_loop_num(cosine_step, cosine_total_step, t_mul):

            loop_num = 0
            cur_end = cosine_total_step
            last_end = 0
            next_end = cosine_total_step

            assert cosine_step >= 0
            assert cosine_total_step > 0
            assert t_mul > 1

            while cosine_step >= cur_end:
                last_end = next_end
                cosine_total_step = int(t_mul * cosine_total_step)
                next_end = last_end + cosine_total_step
                cur_end += cosine_total_step
                loop_num += 1

            return loop_num, last_end, next_end

        learning_rate = learning_rate_base

        if epoch <= warmup_steps and warmup_steps > 0:
            learning_rate = warmup_learning_rate + \
                ((learning_rate_base - warmup_learning_rate) / warmup_steps) * epoch
        else:
            cosine_step = epoch - warmup_steps - 1
            if not cycle:

                if cosine_step < cosine_total_step:
                    learning_rate = (learning_rate_base - least_learnning_rate) * 0.5 * \
                        (math.cos(math.pi * cosine_step /
                                  cosine_total_step) + 1) + least_learnning_rate
                else:
                    learning_rate = least_learnning_rate
            else:

                loop_num, last_end, next_end = get_cycle_loop_num(
                    cosine_step, cosine_total_step, t_mul)
                learning_rate = (learning_rate_base - least_learnning_rate) * (m_mul**loop_num) * 0.5 * \
                    (math.cos(math.pi * (cosine_step - last_end) /
                              (next_end - last_end)) + 1) + least_learnning_rate

        return learning_rate

    def show_lr_curve():
        import matplotlib.pyplot as plt
        lr_list = []
        for index in range(100):
            cur_lr = cosine_decay_with_warmup_schedule(index)
            lr_list.append(cur_lr)
        print(lr_list)

        plt.plot(lr_list)
        plt.show()

    assert learning_rate_base > warmup_learning_rate, "lr after warming-up must be larger"

    # show_lr_curve()

    return LearningRateScheduler(cosine_decay_with_warmup_schedule, verbose=verbose)


def tensor_board(path):
    """
    训练信息记录入tansorboard
    """
    return TensorBoard(log_dir=os.path.join(path, 'log'), write_graph=True)


if __name__ == '__main__':
    learning_rate_cosine_decay_with_warmup_and_cycle()
