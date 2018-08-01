# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.util.Logger import pprint_logger, Logger
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


def load_samsung():
    path = "C:/Users/demetoir/PycharmProjects/MLtools/data/samsung_contest/maindata/data.csv"
    df = pd.read_csv(path, encoding='euc-kr')


    print(df.info())
    print(df.head())



    pass


@deco_timeit
def main():
    load_samsung()
