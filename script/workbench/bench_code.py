# -*- coding:utf-8 -*-
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from script.sklearn_like_toolkit.FeatureEngineerTools import FeatureEngineerTools
from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit

# print(built-in function) is not good for logging

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame


# def titanic_plot_all():
#     df_Xs_keys = [
#         'col_00_Age', 'col_01_Cabin', 'col_02_Embarked', 'col_03_Fare',
#         'col_04_Name', 'col_05_Parch', 'col_06_PassengerId', 'col_07_Pclass',
#         'col_08_Sex', 'col_09_SibSp', 'col_11_Ticket'
#     ]
#     df_Ys_key = 'col_10_Survived'
#
#     merge_df = load_merge_set(cache=False)
#
#     cleaner = titanic_null_cleaner(merge_df, df_Xs_keys, df_Ys_key)
#     # cleaner.boilerplate_maker('./titanic_cleaner.py')
#
#     merge_df = cleaner.clean()
#     typecaster = titanic_typecasting(merge_df, df_Xs_keys, df_Ys_key)
#     # typecaster.boilerplate_maker('./titanic_typecaster.py')
#     merge_df = typecaster.type_cast()
#     transformer = titanic_transformer(merge_df, df_Xs_keys, df_Ys_key)
#     # transformer.boilerplate_maker('./titanic_transformer.py')
#     merge_df = transformer.transform()
#     print(merge_df.info())
#     # print(merge_df.info())
#     # print(merge_df.columns)
#
#     # merge_df.to_csv('./merge_transform.csv')
#     #
#     # print(merge_df.info())
#
#     # plot = PlotTools(show=False, save=True)
#     # plot.plot_all()
#     # plot.count(merge_df, 'col_00_Age_binning', df_Ys_key)
#
#
# def groupby_label(dataset):
#     x = dataset.Ys_index_label
#     # print(x)
#
#     # print(np.bincount(x))
#     # print(np.unique(x))
#
#     label_partials = []
#     for key in np.unique(x):
#         idxs = np.where(x == key)
#         partial = dataset.Xs[idxs]
#         label_partials += [partial]
#
#     return label_partials
#
#
# def titanic_pca():
#     dataset_pack = DatasetPackLoader().load_dataset('titanic', cache=False)
#     train_set = dataset_pack['train']
#     test_set = dataset_pack['test']
#
#     train_set, valid_set = train_set.split((7, 3))
#
#     train_Xs, train_Ys = train_set.full_batch()
#     valid_Xs, valid_Ys = valid_set.full_batch()
#
#     X_0, X_1 = groupby_label(train_set)
#     test_X_0, test_X_1 = groupby_label(valid_set)
#
#     plot = PlotTools(show=False, save=True)
#
#     model = PCA(n_components=2)
#     transformed = model.fit(train_Xs)
#     print(transformed)
#
#     transformed = [model.transform(X_0), model.transform(X_1)]
#     plot.scatter_2d(*transformed, title='pca train', marker_size=10)
#
#     transformed = [model.transform(test_X_0), model.transform(test_X_1)]
#     plot.scatter_2d(*transformed, title='pca test', marker_size=10)
#
#
# def titanic_AE():
#     dataset_pack = DatasetPackLoader().load_dataset('titanic', cache=False)
#     train_set = dataset_pack['train']
#     test_set = dataset_pack['test']
#
#     train_set, valid_set = train_set.split((7, 3))
#
#     train_Xs, train_Ys = train_set.full_batch()
#     valid_Xs, valid_Ys = valid_set.full_batch()
#
#     X_0, X_1 = groupby_label(train_set)
#     test_X_0, test_X_1 = groupby_label(valid_set)
#
#     plot = PlotTools(show=False, save=True)
#
#     vae = AE(learning_rate=0.001, latent_code_size=2, batch_size=32)
#
#     vae.train(train_Xs, epoch=100)
#
#     loss = vae.metric(train_Xs)
#     print(loss)
#     if np.isnan(loss):
#         print('break nan')
#
#     code_0 = vae.code(X_0)
#     code_1 = vae.code(X_1)
#     plot.scatter_2d(code_0, code_1, title=f"ae train", path=f"./matplot/ae_train.png", marker_size=5)
#
#     code_0 = vae.code(test_X_0)
#     code_1 = vae.code(test_X_1)
#     plot.scatter_2d(code_0, code_1, title=f"ae valid", path=f"./matplot/ae_valid.png", marker_size=5)
#
#
# def titanic_clf_pack():
#     def to_df(d):
#         df = DF(d).transpose()
#         df['model'] = df.index
#         df = df.reset_index(drop=True)
#
#         # columns = [
#         #     'model',
#         #     'accuracy',
#         #     'roc_auc_score',
#         #     'confusion_matrix',
#         #     'precision_score',
#         #     'recall_score',
#         # ]
#         return df
#
#     def sum_score_pack(scores):
#         total = {}
#         model_names = scores[0].keys()
#
#         for i in range(len(scores)):
#             for model in model_names:
#                 if model not in total:
#                     total[model] = {}
#                 for k, v in scores[i][model].items():
#                     if k not in total[model]:
#                         total[model][k] = []
#                     total[model][k] += [v]
#
#         keys = [
#             'accuracy',
#             'roc_auc_score',
#             'precision_score',
#             'recall_score',
#         ]
#
#         for model in model_names:
#
#             for k in keys:
#                 total[model][k + '_mean'] = np.mean(total[model][k])
#                 total[model][k + '_std'] = np.std(total[model][k])
#                 total[model][k + '_min'] = np.min(total[model][k])
#                 total[model][k + '_max'] = np.max(total[model][k])
#
#         return total
#
#     def iter_score_pack(clf, train_score_csv_path, valid_score_csv_path, dataset, n_iter=10):
#         train_scores = []
#         valid_scores = []
#
#         for i in range(n_iter):
#             dataset.shuffle()
#             train_set, valid_set = dataset.split((7, 3))
#             train_Xs, train_Ys = train_set.full_batch()
#             valid_Xs, valid_Ys = valid_set.full_batch()
#
#             score = clf.score_pack(train_Xs, train_Ys)
#             score_df = to_df(score)
#             score_df.to_csv(train_score_csv_path)
#             train_scores += [score]
#
#             score = clf.score_pack(valid_Xs, valid_Ys)
#             score_df = to_df(score)
#             score_df.to_csv(valid_score_csv_path)
#             valid_scores += [score]
#
#         train_score_df = to_df(sum_score_pack(train_scores))
#         train_score_df.to_csv(train_score_csv_path)
#         valid_score_df = to_df(sum_score_pack(valid_scores))
#         valid_score_df.to_csv(valid_score_csv_path)
#         print(train_score_df)
#         print(valid_score_df)
#         return train_score_df, valid_score_df
#
#     dataset_pack = DatasetPackLoader().load_dataset('titanic', cache=False)
#     origin_train_set = dataset_pack['train']
#     test_set = dataset_pack['test']
#
#     origin_train_set.shuffle()
#     train_set, valid_set = origin_train_set.split((7, 3))
#     train_Xs, train_Ys = train_set.full_batch()
#     valid_Xs, valid_Ys = valid_set.full_batch()
#
#     clf_path = './out/clf_pack.pkl'
#     if not os.path.exists(clf_path):
#         clf_pack = ClassifierPack()
#         clf_pack.drop('skQDAClf')
#         clf_pack.drop('skGaussian_NBClf')
#         clf_pack.drop('skNearestCentroidClf')
#         clf_pack.drop('skMultinomial_NBClf')
#         clf_pack.fit(train_Xs, train_Ys)
#         clf_pack.save(clf_path)
#     clf_pack = ClassifierPack().load(clf_path)
#
#     train_score_csv_path = './out/clf_pack_train_score_pack.csv'
#     valid_score_csv_path = './out/clf_pack_valid_score_pack.csv'
#     iter_score_pack(clf_pack, train_score_csv_path, valid_score_csv_path, origin_train_set)
#
#     clf_path = './out/clf_pack_Hyperopt.pkl'
#     if not os.path.exists(clf_path):
#         clf_pack = ClassifierPack()
#         clf_pack.drop('skQDAClf')
#         clf_pack.drop('skGaussian_NBClf')
#         clf_pack.drop('skNearestCentroidClf')
#         clf_pack.drop('skMultinomial_NBClf')
#         clf_pack.HyperOptSearch(train_Xs, train_Ys, valid_Xs, valid_Ys, 1000, parallel=True)
#         clf_pack.save(clf_path)
#     clf_pack = ClassifierPack().load(clf_path)
#
#     HyperOpt_clf_train_score_csv_path = './out/clf_pack_Hyperopt_train_score_pack.csv'
#     HyperOpt_clf_valid_score_csv_path = './out/clf_pack_Hyperopt_valid_score_pack.csv'
#     iter_score_pack(clf_pack, HyperOpt_clf_train_score_csv_path, HyperOpt_clf_valid_score_csv_path, origin_train_set)
#
#     esm_pack_path = './out/esm_pack_Hyperopt.pkl'
#     esm_pack = clf_pack.to_ensembleClfpack()
#     esm_pack.fit(train_Xs, train_Ys)
#     esm_pack.save(esm_pack_path)
#
#     iter_score_pack(esm_pack, './out/esm_pack_Hyperopt_train_score_pack.csv',
#                     './out/esm_pack_Hyperopt_valid_score_pack.csv', origin_train_set)
#
#
# def test_titanic_dataset():
#     dataset_pack = DatasetPackLoader().load_dataset('titanic', cache=False)
#     train_set = dataset_pack['train']
#     test_set = dataset_pack['test']
#
#     train_set, valid_set = train_set.split((7, 3))
#
#     train_Xs, train_Ys = train_set.full_batch()
#     valid_Xs, valid_Ys = valid_set.full_batch()
#
#     # clf_pack = ClassifierPack(['LightGBMClf', 'skDecisionTreeClf', 'skMLPClf'])
#
#     X_0, X_1 = groupby_label(train_set)
#     test_X_0, test_X_1 = groupby_label(valid_set)
#     # print(X_0, X_1)
#
#     plot = PlotTools(show=False, save=True)
#
#     model = PCA(n_components=2)
#     transformed = model.fit(train_Xs)
#     print(transformed)
#
#     transformed = [model.transform(X_0), model.transform(X_1)]
#     plot.scatter_2d(*transformed, title='pca train', marker_size=10)
#
#     transformed = [model.transform(test_X_0), model.transform(test_X_1)]
#     plot.scatter_2d(*transformed, title='pca test', marker_size=10)
#
#     vae = AE(learning_rate=0.001, latent_code_size=2, batch_size=32)
#
#     vae.train(train_Xs, epoch=100)
#
#     loss = vae.metric(train_Xs)
#     print(loss)
#     if np.isnan(loss):
#         print('break nan')
#
#     code_0 = vae.code(X_0)
#     code_1 = vae.code(X_1)
#     plot.scatter_2d(code_0, code_1, title=f"ae train", path=f"./matplot/ae_train.png", marker_size=5)
#
#     code_0 = vae.code(test_X_0)
#     code_1 = vae.code(test_X_1)
#     plot.scatter_2d(code_0, code_1, title=f"ae valid", path=f"./matplot/ae_valid.png", marker_size=5)
#
#     #
#     # for i in range(100, 400, 1):
#     #     vae.batch_size = 128
#     #     vae.train(train_Xs)
#     #     code_0 = vae.code(X_0)
#     #     code_1 = vae.code(X_1)
#     #
#     #     loss = vae.metric(train_Xs)
#     #     print(loss)
#     #     if np.isnan(loss):
#     #         print('break nan')
#     #         break
#     #
#     #     plot.scatter_2d(code_0, code_1, title=f"train vae epoch = {i}", path=f"./matplot/epoch{i}_train.png")
#     #
#     #     code_0 = vae.code(test_X_0)
#     #     code_1 = vae.code(test_X_1)
#     #     plot.scatter_2d(code_0, code_1, title=f"test vae epoch = {i}", path=f"./matplot/epoch{i}_test.png")
#
#     # for i in range(100, 500, 1):
#     #     vae.train(train_Xs, batch_size=64)
#     #     code_0 = vae.code(X_0)
#     #     code_1 = vae.code(X_1)
#     #
#     #     loss = vae.metric(train_Xs)
#     #     print(loss)
#     #     if np.isnan(loss):
#     #         print('break nan')
#     #         break
#     #
#     #     plot.scatter_2d(code_0, code_1, title=f"vae epoch = {i}", path=f"./matplot/epoch{i}.png")
#
#
# def test_plot_percentage_stack_bar():
#     df_Xs_keys = [
#         'col_00_Age',
#         'col_01_Cabin',
#         'col_02_Embarked',
#         'col_03_Fare',
#         'col_04_Name',
#         'col_05_Parch',
#         'col_06_PassengerId',
#         'col_07_Pclass',
#         'col_08_Sex',
#         'col_09_SibSp',
#         'col_11_Ticket'
#     ]
#     df_Ys_key = 'col_10_Survived'
#
#     merge_df = load_merge_set(cache=False)
#     plot = PlotTools(show=False, save=True)
#     # print(merge_df.info())
#
#     plot.plot_percentage_stack_bar(merge_df, 'col_07_Pclass', 'col_10_Survived')
#
#     pass
#
#
# def test_plot_table():
#     df_Xs_keys = [
#         'col_00_Age',
#         'col_01_Cabin',
#         'col_02_Embarked',
#         'col_03_Fare',
#         'col_04_Name',
#         'col_05_Parch',
#         'col_06_PassengerId',
#         'col_07_Pclass',
#         'col_08_Sex',
#         'col_09_SibSp',
#         'col_11_Ticket'
#     ]
#     df_Ys_key = 'col_10_Survived'
#
#     merge_df = load_merge_set(cache=False)
#     plot = PlotTools(show=False, save=True)
#     # print(merge_df.info())
#
#     df = merge_df.groupby(['col_07_Pclass', df_Ys_key])['col_06_PassengerId'].count()
#     df = DF(df)
#     print(df)
#
#     # print(tabulate(df, tablefmt="grid", headers="keys"))
#
#     # plot.plot_table(df)
#     # plot.plot_percentage_stack_bar(merge_df, 'col_07_Pclass', 'col_10_Survived')
#
#
# def submit_titanic():
#     dataset_pack = DatasetPackLoader().load_dataset('titanic', cache=False)
#     origin_train_set = dataset_pack['train']
#     test_set = dataset_pack['test']
#
#     origin_train_set.shuffle()
#     train_set, valid_set = origin_train_set.split((7, 3))
#     train_Xs, train_Ys = train_set.full_batch()
#     valid_Xs, valid_Ys = valid_set.full_batch()
#     test_Xs = test_set.full_batch(['Xs'])
#
#     clf_path = './out/clf_pack_Hyperopt.pkl'
#     if not os.path.exists(clf_path):
#         clf_pack = ClassifierPack()
#         clf_pack.drop('skQDAClf')
#         clf_pack.drop('skGaussian_NBClf')
#         clf_pack.drop('skNearestCentroidClf')
#         clf_pack.drop('skMultinomial_NBClf')
#         clf_pack.HyperOptSearch(train_Xs, train_Ys, valid_Xs, valid_Ys, 1000, parallel=True)
#         clf_pack.save(clf_path)
#     clf_pack = ClassifierPack().load(clf_path)
#     predict_Y = clf_pack.predict(test_Xs)['skBaggingClf']
#     print(predict_Y.shape)
#     titanic().to_kaggle_submit_csv('./skBaggingClf_submit.csv', predict_Y)
#     predict_Y = clf_pack.predict(test_Xs)['CatBoostClf']
#     titanic().to_kaggle_submit_csv('./CatBoostClf_submit.csv', predict_Y)
#     predict_Y = clf_pack.predict(test_Xs)['skGaussianProcessClf']
#     titanic().to_kaggle_submit_csv('./skGaussianProcessClf_submit.csv', predict_Y)
#
#     # esm_pack_path = './out/esm_pack_Hyperopt.pkl'
#     # esm_pack = clf_pack.to_ensembleClfpack()
#     # esm_pack.fit(train_Xs, train_Ys)
#     # esm_pack.save(esm_pack_path)
#     # predict_Y = esm_pack.predict(test_Xs)['mlxStackingCVClf']
#     # titanic().to_kaggle_submit_csv('./mlxStackingClf_submit.csv', predict_Y)
#     # predict_Y = esm_pack.predict(test_Xs)['FoldingHardVote']
#     # titanic().to_kaggle_submit_csv('./FoldingHardVote_submit.csv', predict_Y)
#
#
# def test_plot():
#     dataset_pack = DatasetPackLoader().load_dataset('titanic', cache=False)
#     train_set = dataset_pack['train']
#
#     df = train_set.to_DataFrame()
#     print(df.info())

def drop_rows(df, row_idxs):
    return df.drop(row_idxs, axis=0)


def load_samsung(path):
    return pd.read_csv(path, encoding='euc-kr', engine='python')


def save_samsung(df, path):
    df.to_csv(path, encoding='euc-kr', index=False)


def add_col_num(df, zfill=2):
    df_cols = df.columns
    new_cols = []
    for i, col in enumerate(df_cols):
        new_cols += ['c' + str(i).zfill(zfill) + '_' + col]
    df.columns = new_cols
    return df


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


def common_label_and_onehot(df, key):
    fe_tool = FeatureEngineerTools()
    unique = list(df[key].unique())
    unique += ['none']

    labeled, labeled_col, enc, dec = fe_tool.to_label(df, key, unique=unique, with_mapper=True)
    df = fe_tool.concat_df(df, labeled)

    onehot_df = fe_tool.to_onehot(df, key, unique=unique)
    df = fe_tool.concat_df(df, onehot_df)

    return df


def c_15(df, key='c15_당사자종별_2당_대분류'):
    df.loc[df[key] == '0', key] = '없음'

    df = common_label_and_onehot(df, key)
    return df


def c_14(df, key='c14_당사자종별_1당_대분류'):
    df = common_label_and_onehot(df, key)
    return df


def c_13(df, key='c13_도로형태'):
    df = common_label_and_onehot(df, key)
    return df


def c_12(df, key='c12_도로형태_대분류'):
    df = common_label_and_onehot(df, key)
    return df


def c_11(df, key="c11_법규위반"):
    df = common_label_and_onehot(df, key)
    return df


def c_10(df, key='c10_사고유형_중분류'):
    df = common_label_and_onehot(df, key)
    return df


def c_09(df, key='c09_사고유형_대분류'):
    df = common_label_and_onehot(df, key)
    return df


def c_08(df, key='c08_발생지시군구'):
    df = common_label_and_onehot(df, key)
    return df


def c_07(df, key='c07_발생지시도'):
    df = common_label_and_onehot(df, key)
    return df


def c_00(df, key='c01_요일'):
    df = common_label_and_onehot(df, key)
    return df


def c_01(df, key='c00_주야'):
    df = common_label_and_onehot(df, key)
    return df


def init_samsung():
    path = "./data/samsung_contest/main/Train_교통사망사고정보(12.1~17.6).csv"
    df = load_samsung(path)

    # train columns to make like test columns
    path = "./data/samsung_contest/test_kor.csv"
    test_df = load_samsung(path)
    test_cols = test_df.columns
    df = DF(df[test_cols])
    df = add_col_num(df, 2)

    # drop null include rows
    cols = [
        'c00_주야',
        'c01_요일',
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
        'c07_발생지시도',
        'c08_발생지시군구',
        'c09_사고유형_대분류',
        'c10_사고유형_중분류',
        'c11_법규위반',
        'c12_도로형태_대분류',
        'c13_도로형태',
        'c14_당사자종별_1당_대분류',
        'c15_당사자종별_2당_대분류'
    ]
    idx = df[df['c15_당사자종별_2당_대분류'].isna()].index
    df = drop_rows(df, idx)
    df = DF(df)

    path = "./data/samsung_contest/data_init.csv"
    save_samsung(df, path)


def samsung_transform(cache=False):
    init_samsung()
    path = "./data/samsung_contest/data_init.csv"
    df = load_samsung(path)

    id_col = 'c.._id'
    path = "./data/samsung_contest/data_tansformed.csv"
    if not os.path.exists(path) or not cache:
        print('transform')
        funcs = [c_15, c_14, c_13, c_12, c_11, c_10, c_09, c_08, c_07, c_01, c_00]
        for func in tqdm(funcs):
            df = func(df)

        df = df.drop_duplicates(keep='first')

        df[id_col] = np.arange(0, len(df))
        save_samsung(df, path)

    df = load_samsung(path)
    reg_cols = [
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
    ]

    label_encoder_cols = []
    for k in df.columns:
        if '_label' in k:
            label_encoder_cols += [k]

    onehot_col = []
    for k in df.columns:
        if '_onehot' in k:
            onehot_col += [k]

    x_cols = reg_cols + onehot_col

    origin_cols = [
        'c00_주야',
        'c01_요일',
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
        'c07_발생지시도',
        'c08_발생지시군구',
        'c09_사고유형_대분류',
        'c10_사고유형_중분류',
        'c11_사고유형',
        'c12_법규위반',
        'c13_도로형태_대분류',
        'c14_도로형태',
        'c15_당사자종별_1당_대분류',
        'c16_당사자종별_2당_대분류',
    ]

    # pprint(label_encoder_cols)
    # pprint(onehot_col)
    # pprint(x_cols)
    # pprint(origin_cols)

    # print(df.info())
    path = "./data/samsung_contest/data_in_progress.csv"
    save_samsung(df, path)

    def random_col_nullify(df, n_iter, ):
        rows = len(df)
        cols = len(df.columns)

        for _ in range(n_iter):
            while True:
                r = np.random.randint(0, rows)
                c = np.random.randint(0, cols)
                if df.loc[r, c] != 'none':
                    df.loc[r, c] = 'none'
                    break

    def duplicated_count(df):
        origin_df = DF(df[origin_cols])
        origin_df['dummy'] = np.zeros(shape=[len(origin_df)])
        pprint(origin_df.info())
        groupby = origin_df.groupby(origin_cols)['dummy'].count()
        # print(groupby)
        print(groupby.value_counts())


def dict_transpose(d):
    return {val: key for key, val in d.items()}


def get_nullify_type():
    path = "./data/samsung_contest/result_kor.csv"
    df = load_samsung(path)

    # cols = df.columns
    cols = ['열', '행', '값']

    # print(df)
    # print(df.info())
    # groupby = df.groupby('열')['행'].count()
    # print(groupby)

    unique = df['열'].unique()
    nullify_type_d = {}
    nullify_type = {}
    for val in unique:
        t = sorted(list(df[df['열'] == val]['행']))
        nullify_type[t.__repr__()] = t
        t = t.__repr__()

        if t in nullify_type_d:
            nullify_type_d[t] += 1
        else:
            nullify_type_d[t] = 1

    nullify_type = sorted(list(nullify_type.values()))

    pprint(nullify_type_d)
    pprint(nullify_type)
    return nullify_type


def col_mapper():
    path = "./data/samsung_contest/test_kor.csv"
    df = load_samsung(path)
    df = add_col_num(df, 2)
    cols = list(df.columns)
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    to_col_name = dict(zip(alpha, cols))
    to_col_alpha = dict_transpose(to_col_name)

    return to_col_name, to_col_alpha


def nullify_df(df: DF, type_, null='none', copy=True):
    if copy:
        df = DF(df)

    to_col_name, to_col_alpha = col_mapper()

    df_cols = df.columns
    for col_alpha in type_:
        col_name = to_col_name[col_alpha]
        if col_name not in df_cols:
            raise ValueError(f"{col_name} not in df columns")

        df[col_name] = null
    return df


def hyp_1(df):
    # 대분류 예측으로 사망자 판단.. 그래프 확인하기

    unique_val = {}
    for a, b in zip(list(df['당사자종별_1당']), list(df['당사자종별_1당_대분류'])):
        x = (a, b)
        if x in unique_val:
            unique_val[x] += 1
        else:
            unique_val[x] = 1
    pprint(unique_val)
    print('check val end')

    unique_val = {}
    for a, b in zip(list(df['당사자종별_2당']), list(df['당사자종별_2당_대분류'])):
        x = (a, b)
        if x in unique_val:
            unique_val[x] += 1
        else:
            unique_val[x] = 1
    pprint(unique_val)
    print('check val end')


def hyp_2():
    # one type per one model

    pass


def hype_3():
    # one mlp to to all type
    pass


def hype_4():
    # autoencoder or gan one model to all type
    pass


def test_nullify():
    samsung_transform(cache=True)

    to_col_name, to_col_alpha = col_mapper()
    types = get_nullify_type()
    for t in types:
        for i in t:
            print(i, to_col_name[i])
    pprint(types)
    t = types[0]

    path = "./data/samsung_contest/data_tansformed.csv"
    df = load_samsung(path)
    df = df[:10]
    pprint(df)
    pprint(df.info())
    df = nullify_df(df, t)
    pprint(df)
    pprint(df.info())


def samsung_plot_all():
    path = "./data/samsung_contest/data_tansformed.csv"
    df = load_samsung(path)

    reg_cols = [
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
    ]

    label_encoder_cols = []
    for k in df.columns:
        if '_label' in k:
            label_encoder_cols += [k]

    onehot_col = []
    for k in df.columns:
        if '_onehot' in k:
            onehot_col += [k]

    x_cols = reg_cols + onehot_col

    origin_cols = [
        'c00_주야',
        'c01_요일',
        'c02_사망자수',
        'c03_사상자수',
        'c04_중상자수',
        'c05_경상자수',
        'c06_부상신고자수',
        'c07_발생지시도',
        'c08_발생지시군구',
        'c09_사고유형_대분류',
        'c10_사고유형_중분류',
        'c11_법규위반',
        'c12_도로형태_대분류',
        'c13_도로형태',
        'c14_당사자종별_1당_대분류',
        'c15_당사자종별_2당_대분류',
    ]

    # pprint(label_encoder_cols)
    # pprint(onehot_col)
    # pprint(x_cols)
    # pprint(origin_cols)

    plot = PlotTools()
    for key in origin_cols:
        # plot.dist(df, key, title=f'dist_{key}')
        plot.count(df, key, title=f'count_{key}')

    for a_key in origin_cols:
        for b_key in origin_cols:
            try:
                plot.count(df, a_key, b_key, title=f'count_{a_key}_groupby_{b_key}')
            except BaseException as e:
                print(a_key, b_key, e)



@deco_timeit
def main():
    samsung_plot_all()

    pass
