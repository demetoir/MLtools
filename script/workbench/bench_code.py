# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from script.data_handler.Base.df_plotterMixIn import DF_PlotTools
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.data_handler.titanic import load_merge_set, titanic_null_cleaner, titanic_typecasting, titanic_transformer
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit

# print(built-in function) is not good for logging
bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame


def titanic_plot_all():
    df_Xs_keys = [
        'col_00_Age', 'col_01_Cabin', 'col_02_Embarked', 'col_03_Fare',
        'col_04_Name', 'col_05_Parch', 'col_06_PassengerId', 'col_07_Pclass',
        'col_08_Sex', 'col_09_SibSp', 'col_11_Ticket'
    ]

    df_Ys_key = 'col_10_Survived'

    merge_df = load_merge_set()
    print(merge_df.info())
    # print(merge_df.keys())

    cleaner = titanic_null_cleaner(merge_df, df_Xs_keys, df_Ys_key)

    # cleaner.plot_all(merge_df, df_Xs_keys, df_Ys_key)

    cleaner.boilerplate_maker('./titanic_cleaner.py')

    merge_df = cleaner.clean()

    typecaster = titanic_typecasting(merge_df, df_Xs_keys, df_Ys_key)
    # typecaster.boilerplate_maker('./titanic_typecaster.py')
    merge_df = typecaster.type_cast()

    transformer = titanic_transformer(merge_df, df_Xs_keys, df_Ys_key)

    # transformer.boilerplate_maker('./titanic_transformer.py')
    merge_df = transformer.transform()

    plot = DF_PlotTools(merge_df, df_Ys_key)
    plot.plot_all()


def test_titanic_dataset():
    dataset_pack = DatasetPackLoader().load_dataset('titanic')
    train_set = dataset_pack['train']
    test_set = dataset_pack['test']
    df = train_set.to_DataFrame()

    # print(df.head())
    print(df.info())
    df.to_csv('./sample_train.csv')
    df = test_set.to_DataFrame()
    df.to_csv('./sample_test.csv')

    train_set, valid_set = train_set.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch()
    valid_Xs, valid_Ys = valid_set.full_batch()

    clf_pack = ClassifierPack()
    clf_pack.fit(train_Xs, train_Ys)
    score = clf_pack.score_pack(train_Xs, train_Ys)
    pprint(score)

    score = clf_pack.score(valid_Xs, valid_Ys)
    pprint(score)


@deco_timeit
def main():
    test_titanic_dataset()
    pass
