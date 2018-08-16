# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from script.data_handler.samsun_contest import path_head, samsung_contest
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit
# print(built-in function) is not good for logging
from script.workbench.samsung_inference import SamsungInference

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series


def print_df_unique(df, key=None):
    if key is None:
        for key in df.columns:
            print(key)
            unique = df[key].unique()
            pprint(unique)
            pprint(df[key].value_counts())
            print(key, len(unique))
    else:
        unique = df[key].unique()
        print(key)
        pprint(unique)
        pprint(len(unique))
        pprint(df[key].value_counts())


def exploit_data():
    dataset = samsung_contest(caching=True)
    dataset.load(path_head)
    trainset = dataset.pack['full_set']
    df = trainset.to_DataFrame()

    origin_cols = [
        # 'c00_주야',
        # 'c01_요일',
        'c02_사망자수',
        'c03_사상자수',
        # 'c04_중상자수',
        # 'c05_경상자수',
        # 'c06_부상신고자수',
        # 'c07_발생지시도',
        # 'c08_발생지시군구',
        # 'c09_사고유형_대분류',
        # 'c10_사고유형_중분류',
        # 'c11_법규위반',
        # 'c12_도로형태_대분류',
        # 'c13_도로형태',
        # 'c14_당사자종별_1당_대분류',
        # 'c15_당사자종별_2당_대분류',
    ]
    cols = []
    print(df.columns)
    for col in df.columns:
        for origin_col in origin_cols:
            if origin_col in col:
                cols += [col]
    cols = sorted(list(set(cols)))

    # def f(df, c1, c2, col_name='c1/c2'):
    #     df = DF(df[[c1, c2]])
    #     df['d'] = [0] * len(df)
    #     a = df.groupby([c1, c2])['d'].count()
    #     a = DF(a)
    #     a[col_name] = a.index
    #     a = a.reset_index(drop=True)
    #     a = a[a['d'] >= 100]
    #     print(a.head(145))

    for col in cols:
        print(col, df[col].value_counts())
        print()

    print(cols)


def test_transform():
    dataset = samsung_contest(caching=False)
    dataset.load(path_head)
    trainset = dataset.pack['full_set']
    df = trainset.to_DataFrame()
    print(list(df.columns))
    x_keys = ['c00_주야', 'c01_요일', 'c02_사망자수', 'c03_사상자수', 'c04_중상자수', 'c05_경상자수', 'c06_부상신고자수', 'c07_발생지시도',
              'c08_발생지시군구',
              # 'c09_사고유형_대분류',
              'c10_사고유형_중분류', 'c11_법규위반', 'c13_도로형태', 'c12_도로형태_대분류', 'c14_당사자종별_1당_대분류',
              'c15_당사자종별_2당_대분류']
    y_keys = ['c09_사고유형_대분류']
    trainset.x_keys = x_keys
    trainset.y_keys = y_keys
    print(df.info())
    print(df.head())

    x, y = trainset.next_batch(10)
    print(x, y)
    print(x.shape, y.shape)

    x, y = trainset.next_batch(50, balanced_class=True)
    print(x, y)
    print(x.shape, y.shape)

    x, y = trainset.full_batch()
    print(trainset.classes)
    print(trainset.n_classes)
    print(trainset.size_group_by_class)
    print(x, y)
    print(x.shape, y.shape)
    print(trainset)

    a_set, b_set = trainset.split()
    print(a_set)
    print(b_set)


@deco_timeit
def main():
    SamsungInference().pipeline()
    pass
