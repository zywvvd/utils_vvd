## 通用工具仓库

> 仓库存放 VVD 常用工具，目前包含：

```python
## 整合常用os操作
from os.path import basename as OS_basename
from os.path import join as OS_join
from os.path import exists as OS_exists
from os.path import isdir as OS_isdir
```

```python
class MyEncoder(json.JSONEncoder):
    """
    自定义序列化方法，解决 TypeError - Object of type xxx is not JSON serializable 错误
    使用方法：
    在json.dump时加入到cls中即可，例如：
    json.dumps(data, cls=MyEncoder) 
    """
```

```python
class Loger_printer():
    """
    日志打印类
    会在控制台与日志同时打印信息    
    """
```

```python
def log_init(log_path):
    """
    initialize logging 
    save the logging object in `config.Parameters.Logging_Object`
    
    after this operation,
    we could save logs with simple orders such as `logging.debug('test debug')` `logging.info('test info')` 
    logging level : debug < info < warning <error < critical
    
    Loger_printer.vvd_logging('test')
    """
```

```python
def dir_check(dir_path):
    """
    check if `dir_path` is a real directory path
    if dir not found, make one
    """
```

```python
def cv_image_show(image,window_name='image show'):
    '''
    show image (for debug)
    press anykey to destory the window 
    
    image: image in numpy 
    window_name: name of the window
    
    image color - bgr
    ''' 
```

```python
def extend_image_channel(image):
    '''
    cv显示三通道图像，本函数将原始图像扩展到三通道
    '''
```

```python
def image_show(image,window_name='image show'):
    '''
    更加鲁棒地显示图像包括二维图像,第三维度为1的图像
    '''
```

```python
def data_show(data):
    '''
    show data in a chart
    '''
```

```python
def cv_rgb_imread(image_path):
    """
    按照RGB顺序使用cv读取图像
    """
```

```python
def dir_exists(dir_path):
    """
    check if dir exists
    """
```

### 使用方法

> 下载后直接加载即可

```shell
from utils_vvd.utils import image_show
```

