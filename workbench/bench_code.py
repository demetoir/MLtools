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


def exp_model_confidence():
    data_pack = DatasetPackLoader().load_dataset("titanic")
    # data_pack.shuffle()
    train_dataset = data_pack.set['train']
    train_dataset.sort()
    full_Xs, full_Ys = train_dataset.full_batch(['Xs', 'Ys'])

    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    dump_path = path_join('.', 'sklearn_pkl_dump')
    clf_pack_path = path_join(dump_path, ClassifierPack.__name__ + '.pkl')
    esm_pack_path = path_join(dump_path, EnsembleClfPack.__name__ + '.pkl')

    if not os.path.exists(clf_pack_path):
        clf_pack = ClassifierPack()
        clf_pack.param_search(train_Xs, train_Ys)
        clf_pack.dump(clf_pack_path)

    clf_pack = ClassifierPack.load(clf_pack_path)
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxLogisticRegression')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('skQDA')

    esm_pack = clf_pack.make_ensembleClfpack()
    esm_pack.fit(train_Xs, train_Ys)

    train_dataset.sort()
    full_Xs, full_Ys = train_dataset.full_batch(['Xs', 'Ys'])

    confidence_result_path = 'confidence_result.csv'
    df = pd.DataFrame()

    for key, val in esm_pack.predict_confidence(full_Xs).items():
        df[key] = val
    pprint(df.head(5))

    difficulty_path = './titanic_difficulty_stat.csv'
    difficulty_stat_df = pd.DataFrame().from_csv(difficulty_path)
    survived = difficulty_stat_df[['Survived']]
    difficulty_df = difficulty_stat_df[['total_difficulty']]
    df = pd.concat([df, difficulty_df, survived], axis=1)

    for key, predict in esm_pack.predict(full_Xs).items():
        if 'Vote' in key:
            df[key + '_score'] = np.equal(predict, survived['Survived'] ).astype(int)

    score = esm_pack.score_pack(full_Xs, full_Ys)
    pprint(score)

    # pprint(df.head(5))

    df.to_csv(confidence_result_path)


def exp_titanic_data_difficulty():
    data_pack = DatasetPackLoader().load_dataset("titanic")
    train_dataset = data_pack.set['train']
    train_dataset.shuffle()
    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

    dump_path = path_join('.', 'sklearn_pkl_dump')
    clf_pack_path = path_join(dump_path, ClassifierPack.__name__ + '.pkl')
    if not os.path.exists(clf_pack_path):
        train_dataset.shuffle()
        train_set, valid_set = train_dataset.split((7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

        clf_pack = ClassifierPack()
        clf_pack.param_search(train_Xs, train_Ys)
        clf_pack.dump(clf_pack_path)

    clf_pack = ClassifierPack.load(clf_pack_path)
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxLogisticRegression')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('skQDA')

    def difficulty_stat(pack, dataset, iter=100):
        dataset.sort()
        full_Xs, full_Ys = dataset.full_batch(['Xs', 'Ys'])

        ret = {}
        for clf_key, clf in pack.pack.items():
            print(clf_key)
            predicts = []
            for _ in trange(iter):
                dataset.shuffle()
                train_set, valid_set = dataset.split((7, 3))
                train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

                clf.fit(train_Xs, train_Ys)
                predict = clf.predict(full_Xs)
                predicts += [predict]

            predicts = np.array(predicts)
            predicts = predicts.transpose()

            stat_dicts = {'mean': [], 'std': []}
            for idx, predict in enumerate(predicts):
                row_stat = np_stat_dict(predict)
                for key in stat_dicts:
                    stat_dicts[key] += [row_stat[key]]

            for key in stat_dicts:
                ret[f'{clf_key}_{key}'] = stat_dicts[key]

        # pprint(ret.keys())
        # for key in ret:
        #     pprint(key, ret[key][:3], len(ret[key]))

        return ret

    result_path = './titanic_difficulty_stat.csv'
    if not os.path.exists(result_path):
        esm_pack = clf_pack.make_ensembleClfpack()
        esm_pack.fit(train_Xs, train_Ys)

        pack_dict = difficulty_stat(clf_pack, train_dataset, iter=100)
        clf_pack_result_df = pd.DataFrame(pack_dict)
        esm_pack_dict = difficulty_stat(esm_pack, train_dataset, iter=100)
        esm_pack_dict_df = pd.DataFrame(esm_pack_dict)

        train_dataset.sort()
        ground_true_df = train_dataset.to_DataFrame(['Survived'])

        result_df = pd.concat([ground_true_df, clf_pack_result_df, esm_pack_dict_df], axis=1)
        result_df.to_csv('./titanic_difficulty_stat.csv')

    df = pd.DataFrame.from_csv(result_path)

    # pprint(list(df.keys()))
    pprint(df.head(5))
    for key in df:
        if 'min' in key or 'max' in key:
            df = df.drop(columns=[key])
    df.to_csv(result_path)

    keys = [
        'skMLP_mean',
        'skMLP_std',
        'skSGD_mean',
        'skSGD_std',
        'skBernoulli_NB_mean',
        'skBernoulli_NB_std',
        'skMultinomial_NB_mean',
        'skMultinomial_NB_std',
        'skDecisionTree_mean',
        'skDecisionTree_std',
        'skRandomForest_mean',
        'skRandomForest_std',
        'skExtraTrees_mean',
        'skExtraTrees_std',
        'skAdaBoost_mean',
        'skAdaBoost_std',
        'skGradientBoosting_mean',
        'skGradientBoosting_std',
        'skKNeighbors_mean',
        'skKNeighbors_std',
        'skLinear_SVC_mean',
        'skLinear_SVC_std',
        'skRBF_SVM_mean',
        'skRBF_SVM_std',
        'skGaussianProcess_mean',
        'skGaussianProcess_std',
        'skBagging_mean',
        'skBagging_std',
        'XGBoost_mean',
        'XGBoost_std',
        'LightGBM_mean',
        'LightGBM_std',
        'CatBoost_mean',
        'CatBoost_std',
        'mlxPerceptronClf_mean',
        'mlxPerceptronClf_std',
        'FoldingHardVote_mean',
        'FoldingHardVote_std',
        'mlxStackingCVClf_mean',
        'mlxStackingCVClf_std',
        'mlxStackingClf_mean',
        'mlxStackingClf_std']
    for key in keys:
        if 'mean' in key:
            df[key] = abs(df['Survived'] - df[key])
    pprint(df.head(5))
    df.to_csv(result_path)

    df['total_difficulty'] = [0] * len(df)

    count = 0.0
    for key in keys:
        if 'mean' in key:
            count += 1
            df['total_difficulty'] += df[key]
    df['total_difficulty'] = df['total_difficulty'] / count
    pprint(df.head(5))
    df.to_csv(result_path)

    # col = clf_name_mean,clf_name_std ..., feature,
    # sort by difficulty
    # use best...


@deco_timeit
def main():
    exp_titanic_data_difficulty()
    exp_model_confidence()
    pass
