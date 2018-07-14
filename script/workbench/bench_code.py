# -*- coding:utf-8 -*-
import pandas as pd
import inspect
import numpy as np

from script.data_handler.HousePricesTransformer import HousePricesTransformer
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.data_handler.HousePrices import load_merge_set, df_Xs_keys, df_Ys_key, null_cleaning, transform_df, \
    train_test_split
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.sklearn_like_toolkit.HyperOpt.HyperOpt import HyperOpt
# print(built-in function) is not good for logging
from script.sklearn_like_toolkit.RegressionPack import RegressionPack
from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
from script.util.misc_util import path_join
from script.util.pandas_util import df_binning
from unit_test.data_handler.test_HousePrices import test_train_test_split
from unit_test.sklearn_like_toolkit.test_RegressionPack import get_reg_data
from unit_test.util.test_numpy_util import test_np_frequency_equal_bins, test_np_width_equal_bins
from unit_test.util.test_pandas_util import test_df_to_onehot_embedding, test_df_minmax_normalize, test_df_binning

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
from pandas import DataFrame as DF
import scipy

NpArr = np.array


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


def get_current_func_name():
    return inspect.stack()[0][3]


# def test_param():
#     data_pack = DatasetPackLoader().load_dataset('titanic')
#     train_Xs, train_Ys, valid_Xs, valid_Ys = get_reg_data()
#     clf_name = 'XGBoostClf'
#
#     def fit_clf(params):
#         pprint(params)
#         train_set = data_pack['train']
#
#         clf_pack = ClassifierPack()
#         clf = clf_pack[clf_name]
#
#         cv = 3
#         scores = []
#         for _ in range(cv):
#             train_set.shuffle()
#             train_set, valid_set = train_set.split()
#             train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
#             valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])
#
#             clf = clf.__class__(**params)
#             clf.fit(train_Xs, train_Ys)
#             score = clf.score(valid_Xs, valid_Ys)
#
#             scores += [score]
#
#         return -np.mean(scores)
#
#     def fit_reg(params):
#         pprint(params)
#
#         clf_pack = RegressionPack()
#         clf = clf_pack[clf_name]
#
#         cv = 3
#         scores = []
#         for _ in range(cv):
#             clf = clf.__class__(**params)
#             clf.fit(train_Xs, train_Ys)
#             score = clf.score(valid_Xs, valid_Ys)
#
#             scores += [score]
#
#         return np.mean(scores)
#
#     clf_pack = ClassifierPack()
#     clf = clf_pack[clf_name]
#     space = clf.HyperOpt_space
#
#     opt = HyperOpt()
#     best = opt.fit_serial(fit_clf, space, 10)
#
#     # pprint(opt.trials)
#     pprint(opt.losses)
#     pprint(opt.result)
#     pprint(opt.opt_info)
#     pprint(opt.best_param)
#     pprint(opt.best_loss)


def is_categorical(df, key):
    col = df[[key]]
    series = df[key]
    value_counts = series.value_counts()
    # pprint(value_counts)
    pprint(key, len(value_counts))

    pass


def np_entropy_base_bins(np_x: NpArr) -> NpArr:
    pass


def test_np_entropy_base_bins():
    pass


@deco_timeit
def test_HousePrices_dataset():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\HousePrices"""
    merge_df = load_merge_set(dataset_path)

    merge_null_clean = null_cleaning(merge_df)

    transformed = transform_df(merge_null_clean)

    # print(transformed.info())

    # train_df, test_df = train_test_split(transformed)
    #
    # train_df.to_csv(path_join(dataset_path, 'transformed_train.csv'), index=False)
    # test_df.to_csv(path_join(dataset_path, 'transformed_test.csv'), index=False)


@deco_timeit
def main():

    # test_np_equal_bins()
    # test_HousePrices_dataset()
    pass
