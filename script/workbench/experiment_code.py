# -*- coding:utf-8 -*-


import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.data_handler.DummyDataset import DummyDataset
from script.data_handler.HousePrices import HousePricesHelper
from script.data_handler.wine_quality import wine_qualityPack
from script.model.sklearn_like_model.AE.CVAE import CVAE, CVAE_MixIn
from script.model.sklearn_like_model.GAN.C_GAN import C_GAN
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.sklearn_like_toolkit.EnsembleClfPack import EnsembleClfPack
from script.sklearn_like_toolkit.FoldingHardVoteClf import FoldingHardVoteClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf, mlxStackingClf
from script.util.Logger import StdoutOnlyLogger, pprint_logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit, deco_save_log
from script.util.misc_util import path_join
from script.util.numpy_utils import np_stat_dict, np_minmax_normalize
########################################################################################################################
# print(built-in function) is not good for logging
from script.util.pandas_util import df_to_onehot_embedding, df_to_np_onehot_embedding
from unit_test.data_handler.test_wine_quality import load_wine_quality_dataset
from unit_test.model.sklearn_like_model.GAN.test_GAN import to_zero_one_encoding
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from tqdm import trange
import tensorflow as tf

NpArr = np.array

bprint = print
logger = StdoutOnlyLogger()
# noinspection PyShadowingBuiltins
print = logger.get_log()
pprint = pprint_logger(print)


#######################################################################################################################
@deco_timeit
@deco_save_log
def exp_stacking_metaclf(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop('mlxMLP')
    clf.drop('mlxAdaline')
    clf.drop('mlxSoftmaxRegressionClf')
    clf.drop('skGaussian_NB')
    clf.drop('skQDA')

    pack = clf.pack
    for key, meta_clf in pack.items():
        if 'mlx' in key:
            continue

        pprint(f'meta clf = {key}')
        stacking = clf.to_stackingClf(meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score = stacking.score(valid_Xs, valid_Ys)
        pprint(f'score {score}')
        # break


@deco_timeit
@deco_save_log
def exp_stackingCV_metaclf(pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop('mlxMLP')
    clf.drop('mlxAdaline')
    clf.drop('mlxSoftmaxRegressionClf')
    clf.drop('skGaussian_NB')
    clf.drop('skQDA')

    pack = clf.pack
    for key, meta_clf in pack.items():
        if 'mlx' in key:
            continue

        pprint(f'meta clf = {key}')
        stacking = clf.to_stackingCVClf(meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score = stacking.score_pack(valid_Xs, valid_Ys)
        pprint(f'score {score}')
        # break


@deco_timeit
@deco_save_log
def exp_titanic_statistic(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf_pack = ClassifierPack()
    clf_pack.drop('mlxMLP')
    clf_pack.drop('mlxAdaline')
    clf_pack.drop('mlxSoftmaxRegressionClf')
    clf_pack.drop('skGaussian_NB')
    clf_pack.drop('skQDA')

    pack = clf_pack.pack
    pprint(f'pack list {pack}')
    meta_clf = pack['skBernoulli_NB']
    pprint(f'metaclf = {meta_clf}')

    clf_pack.fit(train_Xs, train_Ys)
    score_pack = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint('default param clf pack')
    pprint(score_pack)

    clf_pack.param_search(train_Xs, train_Ys)
    score_pack = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint('optimize param clf pack top1')
    pprint(score_pack)

    pack = [clf for k, clf in clf_pack.pack.items() if hasattr(clf, 'get_params')]
    pack1_default = pack
    pack10_default = pack * 10
    pack100_default_ = pack * 100

    pack1_top1 = clf_pack.clone_top_k_tuned(k=1)
    pack1_top1 = [clf for k, clf in pack1_top1.items() if hasattr(clf, 'get_params')]
    pack10_top1 = pack1_top1 * 10
    pack100_top1 = pack1_top1 * 100

    pack1_top5 = clf_pack.clone_top_k_tuned(k=5)
    pack1_top5 = [clf for k, clf in pack1_top5.items() if hasattr(clf, 'get_params')]
    pack10_top5 = pack1_top5 * 10
    pack100_top5 = pack1_top5 * 100

    def voting_stacking_stackingCV(pack, param_type, pack_n, top):
        pprint(f'param_type={param_type}, pack_n={pack_n}, top={top}')

        voting = FoldingHardVoteClf(pack)
        voting.fit(train_Xs, train_Ys)
        score_pack = voting.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to hard voting')
        pprint(score_pack)

        stacking = mlxStackingClf(pack, meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score_pack = stacking.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to stacking')
        pprint(score_pack)

        stackingCV = mlxStackingCVClf(pack, meta_clf)
        stackingCV.fit(train_Xs, train_Xs)
        score_pack = stackingCV.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to stackingCV')
        pprint(score_pack)

    voting_stacking_stackingCV(pack1_default, 'default', 1, None)
    voting_stacking_stackingCV(pack10_default, 'default', 10, None)
    voting_stacking_stackingCV(pack100_default_, 'default', 100, None)
    voting_stacking_stackingCV(pack1_top1, 'optimize', 1, 'top1')
    voting_stacking_stackingCV(pack10_top1, 'optimize', 10, 'top1')
    voting_stacking_stackingCV(pack100_top1, 'optimize', 100, 'top1')
    voting_stacking_stackingCV(pack1_top5, 'optimize', 1, 'top5')
    voting_stacking_stackingCV(pack10_top5, 'optimize', 10, 'top5')
    voting_stacking_stackingCV(pack100_top5, 'optimize', 100, 'top5')


@deco_timeit
@deco_save_log
def exp_titanic_id_static(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    dataset = dataset.set['train']

    ret_dict = {}
    n = 100
    for i in range(n):
        clf_pack = ClassifierPack()
        dataset.shuffle()
        train_set, valid_set = dataset.split((7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        clf_pack.fit(train_Xs, train_Ys)

        dataset.sort()
        full_Xs, full_Ys = dataset.full_batch(['Xs', 'Ys'])
        predict = clf_pack.predict(full_Xs)

        for key in predict:
            if key in ret_dict:
                ret_dict[key] += predict[key] / float(n)
            else:
                ret_dict[key] = predict[key] / float(n)

    import pandas as pd

    df = pd.DataFrame()
    for key in ret_dict:
        df[key] = ret_dict[key]
    for key in dataset.BATCH_KEYS:
        if key in ['Xs', 'Ys']:
            continue
        print(key, type(key))
        df[key] = dataset.full_batch([key])

    df.to_csv('./exp_titianic_id_result.csv', )


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

    clf_pack = ClassifierPack().load(clf_pack_path)
    clf_pack.drop('mlxAdaline')
    clf_pack.drop('mlxLogisticRegression')
    clf_pack.drop('skGaussian_NB')
    clf_pack.drop('mlxMLP')
    clf_pack.drop('skQDA')

    esm_pack = clf_pack.to_ensembleClfpack()
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
            df[key + '_score'] = np.equal(predict, survived['Survived']).astype(int)

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

    clf_pack = ClassifierPack().load(clf_pack_path)
    clf_pack.drop('mlxAdaline')
    clf_pack.drop('mlxLogisticRegression')
    clf_pack.drop('skGaussian_NB')
    clf_pack.drop('mlxMLP')
    clf_pack.drop('skQDA')

    def difficulty_stat(pack, dataset, n=100):
        dataset.sort()
        full_Xs, full_Ys = dataset.full_batch(['Xs', 'Ys'])

        ret = {}
        for clf_key, clf in pack.pack.items():
            print(clf_key)
            predicts = []
            for _ in trange(n):
                dataset.shuffle()
                train_set, valid_set = dataset.split((7, 3))
                train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

                clf.fit(train_Xs, train_Ys)
                predict = clf.predict(full_Xs)
                predicts += [predict]

            predicts = np.array(predicts)
            predicts = predicts.transpose()

            stat_dicts = {
                'mean': [],
                'std': []
            }
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
        esm_pack = clf_pack.to_ensembleClfpack()
        esm_pack.fit(train_Xs, train_Ys)

        pack_dict = difficulty_stat(clf_pack, train_dataset, n=100)
        clf_pack_result_df = pd.DataFrame(pack_dict)
        esm_pack_dict = difficulty_stat(esm_pack, train_dataset, n=100)
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


def titanic_load_merge_set():
    path = os.getcwd()
    merge_set_path = os.path.join(path, "data", "titanic", "merge_set.csv")
    if not os.path.exists(merge_set_path):
        path = os.getcwd()
        train_path = os.path.join(path, "data", "titanic", "train.csv")
        test_path = os.path.join(path, "data", "titanic", "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        merged_df = pd.concat([train_df, test_df])
        merged_df = merged_df.reset_index(drop=True)

        merged_df.to_csv(merge_set_path)
    else:
        merged_df = pd.read_csv(merge_set_path)

    return merged_df


def exp_age_predict_regr():
    def trans_build_predict_age_dataset():
        path = os.getcwd()
        trans_merge_path = os.path.join(path, "data", "titanic", "trans_merge.csv")
        trans_df = pd.read_csv(trans_merge_path)

        merge_set_path = os.path.join(path, 'data', 'titanic', 'merge_set.csv')
        merge_set = pd.read_csv(merge_set_path)
        age = merge_set[['Age']]

        for col in trans_df.columns:
            if 'Unnamed' in col:
                del trans_df[col]
        # trans_df = trans_df.drop(columns=['Survived'])
        trans_df = trans_df.drop(columns=['Age_bucket'])

        trans_df = pd.concat([trans_df, age], axis=1)

        trans_df_with_age = trans_df.query("""not Age.isna()""")
        # pprint(trans_df_with_age.head(1))
        # pprint(trans_df_with_age.info())

        keys = list(trans_df_with_age.keys().values)

        keys.remove('Age')
        # pprint(keys)

        Xs_df = trans_df_with_age[keys]
        Ys_df = trans_df_with_age[['Age']]
        # pprint(list(Ys_df['Age'].unique()))
        # pprint(Xs_df.info())
        # pprint(Xs_df.head(5))
        #
        # pprint(Ys_df.info())
        # pprint(Ys_df.head(5))

        Xs = df_to_np_onehot_embedding(Xs_df)
        # Ys = df_to_np_onehot_embedding(Ys_df)
        Ys = np.array(Ys_df['Age'])
        # pprint(Xs.shape)
        # pprint(Ys.shape)
        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        # pprint(dataset.to_DataFrame().info())

        return dataset

    def build_predict_age_dataset():
        path = os.getcwd()
        trans_merge_path = os.path.join(path, "data", "titanic", "trans_merge.csv")
        trans_df = pd.read_csv(trans_merge_path)

        for col in trans_df.columns:
            if 'Unnamed' in col:
                del trans_df[col]
        trans_df = trans_df.drop(columns=['Survived'])

        trans_df_with_age = trans_df.query("""Age_bucket != 'None' """)
        pprint(trans_df_with_age.head(1))
        pprint(trans_df_with_age.info())

        keys = list(trans_df_with_age.keys().values)

        keys.remove('Age_bucket')
        pprint(keys)

        Xs_df = trans_df_with_age[keys]
        Ys_df = trans_df_with_age[['Age_bucket']]
        pprint(list(Ys_df['Age_bucket'].unique()))
        # pprint(Xs_df.info())
        # pprint(Xs_df.head(5))

        pprint(Ys_df.info())
        pprint(Ys_df.head())

        Xs = df_to_np_onehot_embedding(Xs_df)
        Ys = df_to_np_onehot_embedding(Ys_df)
        # pprint(Xs.shape)
        # pprint(Ys.shape)
        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        pprint(dataset.to_DataFrame().info())

        return dataset

    dataset = trans_build_predict_age_dataset()
    dataset.shuffle()
    train, test = dataset.split((20, 3))
    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = test.full_batch(['Xs', 'Ys'])

    from xgboost import XGBRegressor
    reg = MLPRegressor(hidden_layer_sizes=(500,))
    reg = LGBMRegressor()
    reg = CatBoostRegressor()
    reg = XGBRegressor()
    reg.fit(train_Xs, train_Ys)
    pprint(reg.score(train_Xs, train_Ys))
    pprint(reg.score(test_Xs, test_Ys))
    n_sample = 5
    pprint(reg.predict(train_Xs[:n_sample]), train_Ys[:n_sample])
    pprint(reg.predict(test_Xs[:n_sample]), test_Ys[:n_sample])

    stat = abs(reg.predict(train_Xs) - train_Ys)
    stat = np_stat_dict(stat)
    pprint(stat)

    stat = abs(reg.predict(test_Xs) - test_Ys)
    stat = np_stat_dict(stat)
    pprint(stat)

    # clf_pack = ClassifierPack(['skMLP'])
    # clf_pack.pack['skMLP'].n_classes_ = test_Ys.shape[1]
    # # clf_pack.fit(train_Xs, train_Ys)
    # clf_pack.param_search(train_Xs, train_Ys)
    # pprint('train', clf_pack.score_pack(train_Xs, train_Ys))
    # pprint('test', clf_pack.score_pack(test_Xs, test_Ys))
    # pprint(['19~30',
    #         '30~40',
    #         '50~60',
    #         '1~3',
    #         '12~15',
    #         '3~6',
    #         '15~19',
    #         '6~12',
    #         '40~50',
    #         '60~81',
    #         '0~1'])


def main():
    print(exp_titanic_statistic.__name__)
    # exp_stacking_metaclf()
    # exp_voting()
    pass


# def exp_titanic_corr_heatmap():
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#
#     def plot_corr_matrix(data, attr, fig_no):
#         corr = data.corr()
#         # f, ax = plt.subplots(figsize=(11, 9))
#
#         # Generate a custom diverging colormap
#         cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
#         # Draw the heatmap with the mask and correct aspect ratio
#         sns.heatmap(corr, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
#                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
#         plt.show()
#
#     path = 'temp.csv'
#     if not os.path.exists(path):
#         df = build_transform(titanic_load_merge_set())
#         df.to_csv(path, index=False)
#         pprint(df.info())
#
#     df = pd.read_csv(path, index_col=False)
#     df = df.query('not Survived.isna()')
#     pprint(df.info())
#
#     df = df_to_onehot_embedding(df)
#     pprint(df.info())
#
#     corr = df.corr()
#     # pprint(corr)
#     pprint(list(corr['Survived_0.0'].items()))
#     pprint(list(corr['Survived_1.0'].items()))
#     # plot_corr_matrix(df, df.keys(), 3)


def exp_C_GAN_with_titanic():
    def show(data):
        pass

    pass
    #
    data_size = 4000
    zero_one_rate = 0.5
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train, valid = train_set.split()

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])
    #
    print(train_Xs.shape, train_Ys.shape)
    # path = 'C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_02-33-02'
    # if not os.path.exists(path):
    gan = C_GAN(learning_rate=0.001, n_noise=32, loss_type='GAN', with_clipping=True, clipping=.15)
    gan.train(train_Xs, train_Ys, epoch=1000)
    # path = gan.save()
    # print(path)

    # path = "C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_03-06-38"
    # gan = C_GAN().load(path)

    metric = gan.metric(train_Xs, train_Ys)
    pprint(metric)

    Ys_gen = [[1, 0] for _ in range(int(data_size * zero_one_rate))] \
             + [[0, 1] for _ in range(int(data_size * (1 - zero_one_rate)))]
    Ys_gen = np.array(Ys_gen)

    Xs_gen = gan.generate(1)
    Xs_gen = to_zero_one_encoding(Xs_gen)
    pprint(Xs_gen)
    pprint(Xs_gen.shape)

    # plot_1d(train_Xs[:1])
    # plot_1d(Xs_gen)
    # plt.plot(train_Xs[:1])
    # plt.show()

    # Xs_merge = np.concatenate([Xs_gen, train_Xs], axis=0)
    # Ys_merge = np.concatenate([Ys_gen, train_Ys], axis=0)
    # clf_pack = ClassifierPack()
    # # clf_pack.drop_clf('skQDA')
    # clf_pack.drop_clf('skGaussian_NB')
    # clf_pack.drop_clf('mlxSoftmaxRegressionClf')
    # clf_pack.drop_clf('mlxPerceptronClf')
    # clf_pack.drop_clf('mlxMLP')
    # clf_pack.drop_clf('mlxLogisticRegression')
    # clf_pack.drop_clf('mlxAdaline')
    # clf_pack.drop_clf('skLinear_SVC')
    # clf_pack.drop_clf('skSGD')
    # clf_pack.drop_clf('skRBF_SVM')
    # clf_pack.drop_clf('skMultinomial_NB')
    # clf_pack.fit(Xs_merge, Ys_merge)
    #
    # score = clf_pack.score(Xs_merge, Ys_merge)
    # pprint(score)
    #
    # score = clf_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = clf_pack.score(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = clf_pack.score(valid_Xs, valid_Ys)
    # pprint(score)

    # esm_pack = clf_pack.to_ensembleClfpack()
    # esm_pack.fit(Xs_merge, Ys_merge)
    #
    # score = esm_pack.score_pack(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = esm_pack.score_pack(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = esm_pack.score_pack(valid_Xs, valid_Ys)
    # pprint(score)


def exp_CVAE_with_titanic():
    data_size = 4000
    zero_one_rate = 0.5
    datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train, valid = train_set.split()

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])

    # path = 'C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_02-33-02'
    # if not os.path.exists(path):
    cvae = CVAE(learning_rate=0.1, latent_code_size=32, verbose=20, KL_D_rate=.05)
    cvae.train(train_Xs, train_Ys, epoch=1)
    # path = gan.save()
    # print(path)

    # path = "C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\instance\\demetoir_C_GAN_2018-06-29_03-06-38"
    # gan = C_GAN().load(path)

    metric = cvae.metric(train_Xs, train_Ys)
    pprint(metric)

    Xs_gen = cvae.recon(train_Xs, train_Ys)

    # plot_1d(Xs_gen[:1])
    # plot_1d(train_Xs[:1])

    #
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = np.concatenate([Xs_gen, cvae.recon(train_Xs, train_Ys)], axis=0)
    # Xs_gen = to_zero_one_encoding(Xs_gen)
    # pprint(Xs_gen)
    # pprint(Xs_gen.shape)
    # Ys_gen = np.concatenate([train_Ys, train_Ys, train_Ys, train_Ys, train_Ys], axis=0)

    # Xs_merge = np.concatenate([Xs_gen, train_Xs], axis=0)
    # Ys_merge = np.concatenate([Ys_gen, train_Ys], axis=0)
    # clf_pack = ClassifierPack()
    # # clf_pack.drop_clf('skQDA')
    # clf_pack.drop_clf('skGaussian_NB')
    # clf_pack.drop_clf('mlxSoftmaxRegressionClf')
    # clf_pack.drop_clf('mlxPerceptronClf')
    # clf_pack.drop_clf('mlxMLP')
    # clf_pack.drop_clf('mlxLogisticRegression')
    # clf_pack.drop_clf('mlxAdaline')
    # clf_pack.drop_clf('skLinear_SVC')
    # clf_pack.drop_clf('skSGD')
    # clf_pack.drop_clf('skRBF_SVM')
    # clf_pack.drop_clf('skMultinomial_NB')
    # clf_pack.fit(Xs_gen, Ys_gen)
    #
    # score = clf_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = clf_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = clf_pack.score(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = clf_pack.score(valid_Xs, valid_Ys)
    # pprint(score)
    #
    # esm_pack = clf_pack.to_ensembleClfpack()
    # esm_pack.fit(Xs_gen, Ys_gen)
    #
    # score = esm_pack.score(Xs_gen, Ys_gen)
    # pprint(score)
    #
    # score = esm_pack.score(train_Xs, train_Ys)
    # pprint(score)
    #
    # score = esm_pack.score(valid_Xs, valid_Ys)
    # pprint(score)
    #
    # test_Xs = test_set.full_batch('Xs')
    # predict = esm_pack.predict(test_Xs)['FoldingHardVote']
    # # predict = clf_pack.predict(test_Xs)['skBagging']
    # pprint(predict)
    # pprint(predict.shape)
    # submit_path = './submit.csv'
    # datapack.to_kaggle_submit_csv(submit_path, predict)


def titanic_submit():
    datapack = DatasetPackLoader().load_dataset('titanic')
    # datapack = DatasetPackLoader().load_dataset('titanic')
    train_set = datapack['train']
    test_set = datapack['test']
    train_set.shuffle()
    train, valid = train_set.split()

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])

    path = './clf_pack.clf'
    if not os.path.exists(path):
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        clf_pack = ClassifierPack()
        clf_pack.gridSearchCV(train_Xs, train_Ys, cv=10)
        clf_pack.dump(path)

    clf_pack = ClassifierPack().load(path)
    # pprint(clf_pack.optimize_result)
    clf_pack.drop('skQDA')
    clf_pack.drop('skGaussian_NB')
    clf_pack.drop('mlxSoftmaxRegressionClf')
    clf_pack.drop('mlxPerceptronClf')
    clf_pack.drop('mlxMLP')
    clf_pack.drop('mlxLogisticRegression')
    clf_pack.drop('mlxAdaline')
    clf_pack.drop('skLinear_SVC')

    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])
    score = clf_pack.score(train_Xs, train_Ys)
    pprint(score)

    score = clf_pack.score(valid_Xs, valid_Ys)
    pprint(score)
    #
    esm_pack = clf_pack.to_ensembleClfpack()
    train, valid = train_set.split((2, 7))
    train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])

    esm_pack.fit(train_Xs, train_Ys)
    pprint(esm_pack.score_pack(train_Xs, train_Ys))
    pprint(esm_pack.score_pack(valid_Xs, valid_Ys))

    test_Xs = test_set.full_batch(['Xs'])

    predict = esm_pack.predict(test_Xs)['FoldingHardVote']
    # predict = clf_pack.predict(test_Xs)['skBagging']
    pprint(predict)
    pprint(predict.shape)
    submit_path = './submit.csv'
    datapack.to_kaggle_submit_csv(submit_path, predict)

    # clf_pack.dump(path)


class autoOnehot(BaseModel, CVAE_MixIn):
    _params_keys = [
        'batch_size',
        'learning_rate',
        'beta1',
        'L1_norm_lambda',
        'K_average_top_k_loss',
        'code_size',
        'z_size',
        'encoder_net_shapes',
        'decoder_net_shapes',
        'with_noise',
        'noise_intensity',
        'loss_type',
        'KL_D_rate'
    ]

    def __init__(self, batch_size=100, learning_rate=0.01, beta1=0.9, L1_norm_lambda=0.001, cate_code_size=32,
                 gauss_code_size=10,
                 z_size=32, encoder_net_shapes=(512,), decoder_net_shapes=(512,), with_noise=False, noise_intensity=1.,
                 loss_type='VAE', KL_D_rate=1.0, verbose=10):
        BaseModel.__init__(self, verbose)
        CVAE_MixIn.__init__(self)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.L1_norm_lambda = L1_norm_lambda
        self.cate_code_size = cate_code_size
        self.gauss_code_size = gauss_code_size
        self.latent_code_size = cate_code_size + gauss_code_size

        self.z_size = self.latent_code_size

        self.loss_type = loss_type
        self.encoder_net_shapes = encoder_net_shapes
        self.decoder_net_shapes = decoder_net_shapes
        self.with_noise = with_noise
        self.noise_intensity = noise_intensity
        self.KL_D_rate = KL_D_rate

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))

        ret.update(self._build_Ys_input_shape(shapes))

        shapes['zs'] = [None, self.z_size]
        ret.update(self._build_zs_input_shape(shapes))

        shapes['noise'] = [None] + list(ret['X_shape'])
        ret.update(self._build_noise_input_shape(shapes))

        return ret

    def encoder(self, Xs, net_shapes, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(flatten(Xs))

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.cate_code_size + self.gauss_code_size * 2, relu)

        return stack.last_layer

    def decoder(self, zs, net_shapes, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(zs)

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.X_flatten_size, relu)
            # stack.linear(self.X_flatten_size)
            # stack.relu()
            # stack.reshape(self.Xs_shape)

        return stack.last_layer

    def aux_reg(self, Xs_onehot, net_shapes, reuse=False, name='aux_reg'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs_onehot)

            for shape in net_shapes:
                stack.linear_block(shape, sigmoid)

            stack.linear_block(1, sigmoid)

        return stack.last_layer

    def _build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')
        self.noises = placeholder(tf.float32, self.noises_shape, name='noises')

        self.Xs_noised = tf.add(self.Xs, self.noises, name='Xs_noised')
        if self.with_noise:
            Xs = self.Xs_noised
        else:
            Xs = self.Xs

        self.h = self.encoder(Xs, self.encoder_net_shapes)

        self.h_cate = self.h[:, :self.cate_code_size]
        self.h_softmax = tf.nn.softmax(self.h_cate)
        self.h_cate_index = tf.argmax(self.h_softmax, 1)
        self.cate_code = tf.one_hot(self.h_cate_index, self.cate_code_size)

        self.h_gauss = self.h[:, self.cate_code_size:]
        self.h_gauss_mean = self.h_gauss[:, : self.gauss_code_size]

        self.h_gauss_std = self.h_gauss[:, self.gauss_code_size:]
        self.h_gauss_std = tf.nn.softplus(self.h_gauss_std)
        self.gauss_code = self.h_gauss_mean + self.h_gauss_std * tf.random_normal(tf.shape(self.h_gauss_mean), 0, 1,
                                                                                  dtype=tf.float32)

        # self.latent_code = tf.one_hot(tf.arg_max(self.h, 1), self.latent_code_size)
        self.latent_code = concat([self.cate_code, self.gauss_code], axis=1)

        # self.latent_code = self.h

        self.Xs_recon = self.decoder(self.latent_code, self.decoder_net_shapes)
        self.Xs_gen = self.decoder(self.zs, self.decoder_net_shapes, reuse=True)

        net_shapes = (512, 512)
        self.h_aux_reg = self.aux_reg(self.latent_code, net_shapes)

        head = get_scope()
        self.vars_encoder = collect_vars(join_scope(head, 'encoder'))
        self.vars_decoder = collect_vars(join_scope(head, 'decoder'))
        self.vars_aux_reg = collect_vars(join_scope(head, 'aux_reg'))

        self.vars = self.vars_encoder + self.vars_decoder

    def _build_loss_ops(self):
        X = flatten(self.Xs)
        X_out = flatten(self.Xs_recon)

        self.MSE = tf.reduce_sum(tf.squared_difference(X, X_out), axis=1)
        if self.loss_type == 'MSE_only':
            self.loss = self.MSE

        self.step_func = tf.cast(X * self.cate_code_size, tf.int32)
        onehot_label = tf.one_hot(self.step_func, self.cate_code_size)
        self.loss_index = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.h_softmax, labels=onehot_label)
        self.loss_index *= 0.001
        # self.loss += self.loss_index

        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

        self.aux_reg_loss = tf.sqrt(tf.squared_difference(self.h_aux_reg, self.Ys))

        self.__metric_ops = [self.loss, self.aux_reg_loss, self.loss_index]

    def _build_train_ops(self):
        var = self.vars_encoder + self.vars_decoder
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=var)

        var = self.vars_aux_reg + self.vars_encoder
        self.train_aux = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.aux_reg_loss,
                                                                                         var_list=var)

        var = self.vars_encoder
        self.train_index = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss_index,
                                                                                           var_list=var)
        self.__train_ops = [self.train_op, self.train_aux, self.train_index, ]
        # self.__train_ops = [self.train_op]

    def train(self, Xs, Ys, epoch=1, save_interval=None, batch_size=None):
        self._prepare_train(Xs=Xs, Ys=Ys)
        dataset = self.to_dummyDataset(Xs=Xs, Ys=Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))

        for e in trange(epoch):
            dataset.shuffle()

            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                noise = self.get_noises(Xs.shape, self.noise_intensity)

                # train_ops = [self.train_op, self.train_aux]
                train_ops = self.__train_ops
                self.sess.run(train_ops, feed_dict={
                    self._Xs: Xs,
                    self._Ys: Ys,
                    self._noises: noise
                })

                recon = self.sess.run(self.Xs_recon, feed_dict={
                    self._Xs: Xs,
                    self._Ys: Ys,
                    self._noises: noise
                })
                self.sess.run(train_ops, feed_dict={
                    self._Xs: recon,
                    self._Ys: Ys,
                    self._noises: noise
                })

            # metric = self.metric(Xs, Ys)
            # self.log.info(f"e:{e} {metric}")

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._code_ops, {
            self._Xs: Xs,
            self._noises: noise
        })

    def recon(self, Xs):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self._recon_ops, {
            self._Xs: Xs,
            self._noises: noise
        })

    def metric(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        metric_ops = self.__metric_ops
        ae_loss, aux_reg_loss, index_loss = self.get_tf_values(metric_ops,
                                                               {
                                                                   self._Xs: Xs,
                                                                   self.Ys: Ys,
                                                                   self._noises: noise
                                                               })
        return {
            'ae_loss': np.mean(ae_loss),
            'aux_reg_loss': np.mean(aux_reg_loss),
            'index_loss': np.mean(index_loss)
        }

    def generate(self, zs, Ys):
        return self.get_tf_values(self._recon_ops, {
            self._Ys: Ys,
            self._zs: zs
        })

    def predict(self, Xs, Ys):
        noise = self.get_noises(Xs.shape)
        return self.get_tf_values(self.h_aux_reg, {
            self._Xs: Xs,
            self.Ys: Ys,
            self._noises: noise
        })
        pass

    def score(self, Xs, Ys):
        predict = self.predict(Xs, Ys)
        return np.sqrt((predict - Ys) * (predict - Ys))


def test_auto_onehot_encoder():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\HousePrices"""
    merge_df = HousePricesHelper.load_merge_set(dataset_path)

    merge_null_clean = HousePricesHelper.null_cleaning(merge_df)

    Xs = merge_null_clean['col_00_1stFlrSF'][:1400]
    Ys = merge_null_clean['col_70_SalePrice'][:1400]

    Xs = NpArr(Xs)
    Ys = NpArr(Ys)
    Xs = np_minmax_normalize(Xs)
    Ys = np_minmax_normalize(Ys)

    Xs = Xs.reshape(-1, 1)
    Ys = Ys.reshape(-1, 1)

    cate_code_size = 10
    aoe = autoOnehot(gauss_code_size=1, cate_code_size=cate_code_size, loss_type='MSE_only', batch_size=128,
                     learning_rate=0.01,
                     encoder_net_shapes=(512, 256),
                     decoder_net_shapes=(256, 512),
                     # with_noise=True,
                     # noise_intensity=0.00001
                     )
    aoe.train(Xs, Ys, epoch=100)
    metric = aoe.metric(Xs, Ys)
    pprint('metric')
    pprint(metric)

    aoe.batch_size = 128
    aoe.train(Xs, Ys, epoch=200)
    metric = aoe.metric(Xs, Ys)
    pprint('metric')
    pprint(metric)

    aoe.batch_size = 256
    aoe.train(Xs, Ys, epoch=400)
    metric = aoe.metric(Xs, Ys)
    pprint('metric')
    pprint(metric)

    # aoe.batch_size = 512
    # aoe.train(Xs, Ys, epoch=1600)
    # metric = aoe.metric(Xs, Ys)
    # pprint('metric')
    # pprint(metric)

    score = aoe.score(Xs, Ys)
    pprint('score')
    pprint(np.mean(score))
    pprint()

    predict = aoe.predict(Xs, Ys)
    pprint('predict')
    pprint(Ys[10:15])
    pprint(predict[10:15])

    latent = aoe.code(Xs)
    pprint('latent')
    pprint(latent[10:15])

    x_walk = np.arange(0, 1, 0.05).reshape(-1, 1)
    y = np.zeros_like(x_walk)
    latent = aoe.code(x_walk)
    recon = aoe.recon(x_walk)
    pprint('latent walk')
    pprint(x_walk)
    pprint(latent)
    pprint(recon)

    recon = aoe.recon(Xs)
    pprint('recon')
    pprint(Xs[10:15])
    pprint(recon[10:15])

    latent = aoe.code(Xs)[:, :cate_code_size]
    print(latent.shape)
    print(np.bincount(np.argmax(latent, axis=1)))

    pprint(aoe.get_tf_values(aoe.h_cate_index, {
        aoe.Xs: Xs[10:15]
    }))
    pprint(aoe.get_tf_values(aoe.step_func,
                             {
                                 aoe.Xs: np.arange(0, 1, 0.05).reshape(-1, 1)
                             }))


def exp_wine_quality_predict():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)
    dataset_pack.shuffle()
    dataset_pack.split('data', 'train', 'test', rate=(7, 3))
    train_set = dataset_pack['train']
    test_set = dataset_pack['test']
    print(train_set.to_DataFrame().info())

    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])

    import os

    clf_pack_path = './clf_pack.pkl'
    if not os.path.exists(clf_pack_path):
        clf_pack = ClassifierPack(['XGBoostClf'])
        # clf_pack.drop('CatBoostClf')
        # clf_pack.drop('skQDAClf')
        # clf_pack.drop('skNearestCentroidClf')
        # clf_pack.drop('skGaussianProcessClf')
        # clf_pack.drop('skGaussian_NBClf')
        # clf_pack.drop('skGradientBoostingClf')
        # clf_pack.drop('skMultinomial_NBClf')
        # clf_pack.drop('skPassiveAggressiveClf')

        # clf_pack.fit(train_Xs, train_Ys)
        clf_pack.HyperOptSearch(train_Xs, train_Ys, 200, parallel=False)

        clf_pack.dump(clf_pack_path)
    else:
        clf_pack = ClassifierPack().load(clf_pack_path)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(train_Xs, train_Ys)

    score = clf_pack.score(train_Xs, train_Ys)
    pprint(score)

    score = clf_pack.score(test_Xs, test_Ys)
    pprint(score)

    # proba = clf_pack.predict_proba(train_Xs[:5])
    # print(proba)

    smaple_X = test_Xs[10:15]
    sample_y = test_Ys[10:15]

    proba = clf_pack.predict_proba(smaple_X)
    print(proba)
    predict = clf_pack.predict(smaple_X)
    print(predict)
    print(sample_y)

    # score = clf_pack.score_pack(train_Xs, train_Ys)
    # pprint(score)
    #
    score = clf_pack.score_pack(test_Xs, test_Ys)
    pprint(score)
    matrix = score['XGBoostClf']['confusion_matrix']
    pprint(matrix / len(test_Xs))


def exp_data_aug_VAE_wine_quality():
    rets = load_wine_quality_dataset()
    Xs, Ys, test_Xs, test_Ys, sample_xs, sample_Ys = rets

    CVAE_path = './CVAE.pkl'
    if not os.path.exists(CVAE_path):
        cvae = CVAE(loss_type='MSE_only', learning_rate=0.01, latent_code_size=5,
                    decoder_net_shapes=(128, 128), encoder_net_shapes=(128, 128), batch_size=512)
        cvae.train(Xs, Ys, epoch=1600)
        cvae.save(CVAE_path)
    else:
        cvae = CVAE().load_meta(CVAE_path)

    metric = cvae.metric(Xs, Ys)
    metric = np.mean(metric)
    print(metric)

    def train_doubling_batch_size(tf_model, Xs, Ys, epoch=100, iter_=3):
        for _ in range(iter_):
            tf_model.train(Xs, Ys, epoch=epoch)
            metric = tf_model.metric(Xs, Ys)
            metric = np.mean(metric)
            print(metric)
            tf_model.batch_size *= 2
            epoch *= 2

    recon = cvae.recon(sample_xs, sample_Ys)
    print(sample_xs)
    pprint(recon)

    gen_labels_1 = np.array([[0, 1] for _ in range(2000)])
    gen_x_1 = cvae.generate(gen_labels_1)
    merged_Xs = np.concatenate([Xs, gen_x_1])
    merged_Ys = np.concatenate([Ys, gen_labels_1])

    gen_labels_0 = np.array([[1, 0] for _ in range(2000)])
    gen_x_0 = cvae.generate(gen_labels_1)
    gen_only_x = np.concatenate([gen_x_0, gen_x_1])
    gen_only_labels = np.concatenate([gen_labels_0, gen_labels_1])

    high = 0.02
    low = -0.02
    noised_Xs = np.concatenate([
        Xs,
        Xs + np.random.uniform(low, high, size=Xs.shape),
        Xs + np.random.uniform(low, high, size=Xs.shape),
        Xs + np.random.uniform(low, high, size=Xs.shape),
    ])
    noised_Ys = np.concatenate([Ys, Ys, Ys, Ys])

    def print_score_info(clf_pack):
        merge_score = clf_pack.score(gen_only_x, gen_only_labels)
        train_score = clf_pack.score(Xs, Ys)
        test_score = clf_pack.score(test_Xs, test_Ys)
        noised_score = clf_pack.score(noised_Xs, noised_Ys)
        print('aug score')
        pprint(f' train_score : {train_score}')
        pprint(f'merge_score : {merge_score}')
        pprint(f'test_score : {test_score}')
        pprint(f'noised_score : {noised_score}')

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(merged_Xs, merged_Ys)
    print('fit merge')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(Xs, Ys)
    print('fit origin')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(gen_only_x, gen_only_labels)
    print('fit gen_only')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(noised_Xs, noised_Ys)
    print('fit noised')
    print_score_info(clf_pack)


def exp_data_aug_GAN_wine_quality():
    rets = load_wine_quality_dataset()
    train_Xs, train_Ys, test_Xs, test_Ys, sample_xs, sample_Ys = rets

    model = C_GAN
    GAN_path = f'./{model.__name__}.pkl'
    if not os.path.exists(GAN_path):
        c_GAN = model(loss_type='WGAN', learning_rate=0.001, batch_size=256, G_net_shape=(256, 256),
                      D_net_shape=(64, 64))
        c_GAN.train(train_Xs, train_Ys, epoch=200)
        c_GAN.save(GAN_path)
    else:
        c_GAN = model().load_meta(GAN_path)

    metric = c_GAN.metric(train_Xs, train_Ys)
    print(metric)

    recon = c_GAN.recon(sample_xs, sample_Ys)
    print(sample_xs)
    pprint(recon)

    gen_labels_1 = np.array([[0, 1] for _ in range(2000)])
    gen_x_1 = c_GAN.generate(gen_labels_1)
    merged_Xs = np.concatenate([train_Xs, gen_x_1])
    merged_Ys = np.concatenate([train_Ys, gen_labels_1])

    gen_labels_0 = np.array([[1, 0] for _ in range(2000)])
    gen_x_0 = c_GAN.generate(gen_labels_1)
    gen_only_x = np.concatenate([gen_x_0, gen_x_1])
    gen_only_labels = np.concatenate([gen_labels_0, gen_labels_1])

    high = 0.02
    low = -0.02
    noised_Xs = np.concatenate([
        train_Xs,
        train_Xs + np.random.uniform(low, high, size=train_Xs.shape),
        train_Xs + np.random.uniform(low, high, size=train_Xs.shape),
        train_Xs + np.random.uniform(low, high, size=train_Xs.shape),
    ])
    noised_Ys = np.concatenate([train_Ys, train_Ys, train_Ys, train_Ys])

    def print_score_info(clf_pack):
        merge_score = clf_pack.score(gen_only_x, gen_only_labels)
        train_score = clf_pack.score(train_Xs, train_Ys)
        test_score = clf_pack.score(test_Xs, test_Ys)
        noised_score = clf_pack.score(noised_Xs, noised_Ys)
        print('aug score')
        pprint(f' train_score : {train_score}')
        pprint(f'merge_score : {merge_score}')
        pprint(f'test_score : {test_score}')
        pprint(f'noised_score : {noised_score}')

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(merged_Xs, merged_Ys)
    print('fit merge')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(train_Xs, train_Ys)
    print('fit origin')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(gen_only_x, gen_only_labels)
    print('fit gen_only')
    print_score_info(clf_pack)

    clf_pack = ClassifierPack(['XGBoostClf'])
    clf_pack.fit(noised_Xs, noised_Ys)
    print('fit noised')
    print_score_info(clf_pack)


def groupby_label(dataset):
    # x = dataset_pack['data'].Ys_onehot_label
    x = dataset.Ys_index_label

    print(np.bincount(x))
    print(np.unique(x))

    label_partials = []
    for key in np.unique(x):
        idxs = np.where(x == key)
        partial = dataset.x[idxs]
        label_partials += [partial]

    return label_partials


def exp_wine_quality_pca():
    df_Xs_keys = [
        'col_0_fixed_acidity', 'col_1_volatile_acidity', 'col_2_citric_acid',
        'col_3_residual_sugar', 'col_4_chlorides', 'col_5_free_sulfur_dioxide',
        'col_6_total_sulfur_dioxide', 'col_7_density', 'col_8_pH',
        'col_9_sulphates', 'col_10_alcohol', 'col_12_color'
    ]
    df_Ys_key = 'col_11_quality'

    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\winequality"""
    dataset_pack = wine_qualityPack().load(dataset_path)
    dataset_pack.shuffle()
    full_Xs, full_Ys = dataset_pack['data'].full_batch(['Xs', 'Ys'])

    dataset_pack.split('data', 'train', 'test', rate=(7, 3))
    train_set = dataset_pack['train']
    test_set = dataset_pack['test']

    Xs, Ys = train_set.full_batch(['Xs', 'Ys'])
    test_Xs, test_Ys = test_set.full_batch(['Xs', 'Ys'])
    sample_xs = Xs[:5]
    sample_Ys = Ys[:5]

    label_partials = groupby_label(dataset_pack['data'])

    model = PCA(n_components=2)
    transformed = model.fit(full_Xs)
    print(transformed)

    transformed = []
    for partial in label_partials:
        pca_partial = model.transform(partial)
        transformed += [pca_partial]

    pprint(transformed)
    plot = PlotTools(show=True, save=False)
    plot.scatter_2d(*transformed)
