#
# image calssification 网络回调函数
#

import os
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau


class ModelCheckpointAfter(ModelCheckpoint):
    """
    每个epoch结束保存模型
    """
    def __init__(self, epoch, steps, filepath, monitor='acc', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
        self.after_epoch = epoch
        self.steps = steps
        self.index = 1

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.after_epoch:
            super().on_epoch_end(self.index, logs)
            self.index+=1
    
    def on_batch_end(self,batch, logs=None):
        print(f'batch: {batch}')
        print(f'logs : {logs}')
        pass

def my_ReduceLROnPlateau(decay_factor, patience, min_lr):
    
    lr_reducer = ReduceLROnPlateau(monitor='loss', factor = decay_factor, cooldown=0, patience= patience, min_lr= min_lr)

    return lr_reducer
  
def model_checkpoint_after(epoch, steps, path, monitor, save_best_only):
    """
    每个epoch结束保存模型
    """
    if not (os.path.exists(path)):
        os.mkdir(path)
    pattern = os.path.join(path, 'epoch-{epoch:03d}-{' + monitor + ':.4f}.h5')

    return ModelCheckpointAfter(epoch, steps, filepath=pattern, monitor=monitor,
                                save_best_only=save_best_only, mode='max')


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
    
  
def learning_rate(step_size, decay, verbose=1):
    """
    学习率衰减
    """
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr
    return LearningRateScheduler(schedule, verbose=verbose)


def tensor_board(path):
    """
    训练信息记录入tansorboard
    """
    return TensorBoard(log_dir=os.path.join(path, 'log'), write_graph=True)
