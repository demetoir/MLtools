# -*- coding:utf-8 -*-
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.util.Logger import pprint_logger, Logger
import numpy as np
from script.util.deco import deco_timeit

########################################################################################################################
# print(built-in function) is not good for logging

bprint = print
logger = Logger('bench_code', level='INFO', )
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
def main():
    data_pack = DatasetPackLoader().load_dataset("titanic")
    train_dataset = data_pack.set['train']
    train_dataset.shuffle()
    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    path = './test_wrapper_pack_grid_search.pkl'
    clf_pack = ClassifierPack()
    clf_pack.fit(train_Xs, train_Ys)
    clf_pack.gridSearchCV(train_Xs, train_Ys)
    score = clf_pack.score_pack(train_Xs, train_Ys)
    pprint(score)
    clf_pack.dump(path)

    clf_pack = ClassifierPack().load(path)
    score = clf_pack.score_pack(train_Xs, train_Ys)
    pprint(score)
    score = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint(score)

    score = clf_pack.score(valid_Xs, valid_Ys)
    pprint(score)
    result = clf_pack.optimize_result
    pprint(result)


    pass
