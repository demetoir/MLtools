# -*- coding:utf-8 -*-
import inspect
import numpy as np
from script.data_handler.HousePrices import HousePricesHelper
# print(built-in function) is not good for logging
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit
from script.util.misc_util import path_join
from script.workbench.experiment_code import test_auto_onehot_encoder

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)

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


@deco_timeit
def test_HousePrices_dataset():
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\HousePrices"""
    merge_df = HousePricesHelper.load_merge_set(dataset_path)

    merge_null_clean = HousePricesHelper.cleaning(merge_df)

    transformed = HousePricesHelper.transform(merge_null_clean)
    # print(transformed.info())

    train_df, test_df = HousePricesHelper.train_test_split(transformed)

    train_df.to_csv(path_join(dataset_path, 'transformed_train.csv'), index=False)
    test_df.to_csv(path_join(dataset_path, 'transformed_test.csv'), index=False)


@deco_timeit
def main():
    test_auto_onehot_encoder()
    # test_HousePrices_dataset()

    # test_np_equal_bins()
    # test_HousePrices_dataset()
    pass
