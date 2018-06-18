# -*- coding:utf-8 -*-
from pandas._libs.parsers import k

from sklearn_like_toolkit.ClassifierPack import ClassifierPack
from data_handler.DatasetPackLoader import DatasetPackLoader
from sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf, mlxStackingClf
from util.Logger import pprint_logger, Logger
import numpy as np
import pandas as pd
from tqdm import trange

from util.deco import deco_timeit, deco_save_log
from util.misc_util import path_join

########################################################################################################################
# print(built-in function) is not good for logging


bprint = print
logger = Logger('bench_code', level='INFO')
print = logger.info
pprint = pprint_logger(print)


#######################################################################################################################


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


def finger_print(size, head='_'):
    ret = head
    h_list = [c for c in '01234556789qwertyuiopasdfghjklzxcvbnm']
    for i in range(size):
        ret += np.random.choice(h_list)
    return ret


@deco_timeit
def test_transform_titanic():
    data_pack = DatasetPackLoader().load_dataset("titanic")
    data_pack.shuffle()
    train_set, valid_set = data_pack.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    def to_df(dataset):
        df = pd.DataFrame()
        for key in dataset.data:
            df[key] = dataset.data[key]
        return df

    clf_pack = ClassifierPack(['LightGBM'])
    clf_pack.fit(train_Xs, train_Ys)
    score_pack = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint(score_pack)

    pass


def main():
    test_transform_titanic()
    pass
