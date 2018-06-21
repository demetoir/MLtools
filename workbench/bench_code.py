# -*- coding:utf-8 -*-
from pandas._libs.parsers import k
import os
from sklearn_like_toolkit.ClassifierPack import ClassifierPack
from data_handler.DatasetPackLoader import DatasetPackLoader
from sklearn_like_toolkit.EnsembleClfPack import EnsembleClfPack
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


def np_stat_dict(a):
    a = np.array(a)
    return {
        'min': np.round(a.min(), decimals=4),
        'mean': np.round(a.mean(), decimals=4),
        'max': np.round(a.max(), decimals=4),
        'std': np.round(a.std(), decimals=4),
        'count': len(a),
    }


def print_score_statistic(clf, scores):
    stat = np_stat_dict(scores)
    pprint(f"""{clf}, {stat}""")

def model_confidence():
    data_pack = DatasetPackLoader().load_dataset("titanic")

    data_pack.shuffle()
    train_dataset = data_pack.set['train']
    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    dump_path = path_join('.', 'sklearn_pkl_dump')
    clf_pack_path = path_join(dump_path, ClassifierPack.__name__ + '.pkl')
    esm_pack_path = path_join(dump_path, EnsembleClfPack.__name__ + '.pkl')

    if not os.path.exists(dump_path):
        clf_pack = ClassifierPack()
        clf_pack.param_search(train_Xs, train_Ys)
        clf_pack.dump(clf_pack_path)

    clf_pack = ClassifierPack.load(clf_pack_path)
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxLogisticRegression')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('skQDA')

    score = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint(score)
    proba = clf_pack.predict_proba(valid_Xs[:5])
    pprint(proba)

    import matplotlib.pyplot as plt

    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    def plot_confidence(confidence, title):
        plt.title(title)
        bin_size = 50
        plt.hist(confidence, bins=[i / bin_size for i in range(bin_size + 1)])
        plt.show()

    # confidence = clf_pack.predict_confidence(valid_Xs)
    # for key, val in confidence.items():
    #     plot_confidence(val, key)

    voting_score = []
    stacking_score = []
    stackingCV_score = []

    for i in range(100):
        data_pack.shuffle()
        train_dataset = data_pack.set['train']
        train_set, valid_set = train_dataset.split((7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

        esm_pack = clf_pack.make_ensembleClfpack()
        esm_pack.fit(train_Xs, train_Ys)
        score = esm_pack.score(valid_Xs, valid_Ys)
        pprint(score)
        # esm_pack.dump(esm_pack_path)

        voting_score += [score['FoldingHardVote']]
        stacking_score += [score['mlxStackingCVClf']]
        stackingCV_score += [score['mlxStackingClf']]

        print_score_statistic('voting', voting_score)
        print_score_statistic('stacking', stacking_score)
        print_score_statistic('stackingCV', stackingCV_score)

        # confidence = esm_pack.predict_confidence(valid_Xs)
        # for key, val in confidence.items():
        #     plot_confidence(val, key)


def titanic_data_difficulty():
    data_pack = DatasetPackLoader().load_dataset("titanic")

    data_pack.shuffle()
    train_dataset = data_pack.set['train']
    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    dump_path = path_join('.', 'sklearn_pkl_dump')
    clf_pack_path = path_join(dump_path, ClassifierPack.__name__ + '.pkl')
    esm_pack_path = path_join(dump_path, EnsembleClfPack.__name__ + '.pkl')

    if not os.path.exists(dump_path):
        clf_pack = ClassifierPack()
        clf_pack.param_search(train_Xs, train_Ys)
        clf_pack.dump(clf_pack_path)

    clf_pack = ClassifierPack.load(clf_pack_path)
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxLogisticRegression')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('skQDA')

    score = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint(score)
    proba = clf_pack.predict_proba(valid_Xs[:5])
    pprint(proba)

    esm_pack = clf_pack.make_ensembleClfpack()
    esm_pack.fit(train_Xs, train_Ys)



    result = pd.DataFrame()
    result.to_csv('./titanic_difficulty.csv')



@deco_timeit
def main():
    pass


