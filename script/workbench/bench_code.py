# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.data_handler.TGS_salt import TGS_salt
from script.model.sklearn_like_model.net_structure.Base_net_structure import Base_net_structure
from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.Stacker import Stacker
from script.util.deco import deco_timeit
# print(built-in function) is not good for logging
from unit_test.model.sklearn_like_model.net_structure.test_FusionNetStructure import test_FusionNetStructure
from unit_test.model.sklearn_like_model.net_structure.test_InceptionStructure import test_InceptionStructure

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series
plot = PlotTools(save=True, show=False)


# plot = PlotTools(save=False, show=True)


def is_white_image(image):
    if np.mean(image) == 255:
        return True
    else:
        return False


def is_black_image(image):
    if np.mean(image) == 0:
        return True
    else:
        return False


def test_empty_mask_clf():
    data_pack = TGS_salt()
    data_pack.load('./data/TGS_salt')
    train_set = data_pack['train']
    test_set = data_pack['test']
    print(train_set.keys)
    train_set.x_keys = ['image']
    train_set.y_keys = ['empty_mask']

    a = train_set.next_batch(10, batch_keys=['image', 'mask', 'empty_mask'], out_type='np_dict')
    image = a['image']
    mask = a['mask']
    y = a['empty_mask']
    plot.plot_image_tile(np.concatenate([image, mask]).reshape([20, 101, 101, 1]), title='empty_mask')

    # cnn = model()
    # for e in range(epoch):
    #     cnn.train(x, y)
    #
    #     metric = cnn.score(x, y)
    #     print(metric)


class empty_mask_clf:
    pass


def test_code_generator():
    import test_mode

    pprint(test_mode.__dict__)
    except_list = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__cached__',
                   '__builtins__']

    pprint(test_mode.__dict__.keys())
    keys = [key for key in test_mode.__dict__.keys() if key not in except_list]
    pprint(keys)

    d = {key: test_mode.__dict__[key] for key in keys}
    pprint(d)

    for k, v in d.items():
        print(k, type(v))


class ExperimentResult:
    def __init__(self):
        self.elapse_time = None
        self.start_time = None
        self.finish_time = None
        self.id = None
        self.model_name = None
        self.epoch = None
        self.dataset = None
        self.git_version = None
        self.machine_info = None

    def __str__(self):
        pass


def get_platform_info():
    import platform

    info = {
        'architecture': platform.architecture(),
        'machine': platform.machine(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'system': platform.system(),
        'version': platform.version(),
        'uname': platform.uname(),
        'win32_ver': platform.win32_ver(),
        'java_ver': platform.java_ver(),
        'dist': platform.dist(),
        'linux_distribution': platform.linux_distribution(),
        'mac_ver': platform.mac_ver(),
    }
    return info


@deco_timeit
def main():
    import platform

    pprint(get_platform_info())

    # test_InceptionStructure()
    pass

    # test_empty_mask_clf()
