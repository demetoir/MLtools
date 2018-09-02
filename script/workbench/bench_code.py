# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.data_handler.TGS_salt import TGS_salt
from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
# print(built-in function) is not good for logging
from slackbot.MLSlackBot import MLSlackBot

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


class empty_mask_clf:
    pass


@deco_timeit
def main():
    pass
