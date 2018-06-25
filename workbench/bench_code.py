# -*- coding:utf-8 -*-
from mlxtend.classifier import MultiLayerPerceptron
from pandas._libs.parsers import k
import os

from data_handler.DummyDataset import DummyDataset
from data_handler.titanic import trans_age, df_to_np_dict, df_to_np_onehot_embedding, build_transform
from data_handler.DatasetPackLoader import DatasetPackLoader
from sklearn_like_toolkit.ClassifierPack import ClassifierPack
from sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostClf
from sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxMLPClf
from sklearn_like_toolkit.warpper.sklearn_wrapper import skQDA
from sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostClf
from util.Logger import pprint_logger, Logger
import numpy as np
import pandas as pd
from tqdm import trange

from util.deco import deco_timeit, deco_save_log
from util.misc_util import path_join

########################################################################################################################
# print(built-in function) is not good for logging
from util.numpy_utils import np_stat_dict, NP_ARRAY_TYPE_INDEX, reformat_np_arr

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


def deco_wait_logger(func):
    def _wrapper(*args, **kwargs):
        global logger
        ret = func(*args, **kwargs)
        # logger.file_handler.join()
        # del logger
        return func

    return _wrapper


def load_merge_set():
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

    from sklearn.neural_network.multilayer_perceptron import MLPRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    # reg = MLPRegressor(hidden_layer_sizes=(500,))
    # reg = LGBMRegressor()
    # reg = CatBoostRegressor()
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


def trans_cabin(df):
    df = pd.DataFrame(df['Cabin'])

    cabin_head = df.query('not Cabin.isna()')
    cabin_head = cabin_head['Cabin'].astype(str)
    cabin_head = cabin_head.str.slice(0, 1)

    na = df.query('Cabin.isna()')

    df.loc[na.index, 'Cabin'] = 'None'
    df.loc[cabin_head.index, 'Cabin'] = cabin_head
    return df


def test_trans_cabin():
    merge_set_df = load_merge_set()
    cabin = trans_cabin(merge_set_df)

    pprint(cabin.head(5))
    pprint(cabin.info())


def df_onehot_embedding(df):
    ret = pd.DataFrame({'_idx': [i for i in range(len(df))]})
    for df_key in df.keys():
        # print(df_key)
        np_arr = np.array(df[df_key])
        for unique_key in sorted(df[df_key].unique()):
            # print(unique_key)
            ret[f'{df_key}_{unique_key}'] = np.array(np.where(np_arr == unique_key, 1, 0).reshape([-1, 1]))

    # ret = np.concatenate([v for k, v in ret.items()], axis=1)
    ret = ret.drop(columns=['_idx'])
    return ret


def exp_titanic_corr_heatmap():
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_corr_matrix(data, attr, fig_no):
        corr = data.corr()
        # f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

    path = 'temp.csv'
    if not os.path.exists(path):
        df = build_transform(load_merge_set())
        df.to_csv(path, index=False)
        pprint(df.info())

    df = pd.read_csv(path, index_col=False)
    df = df.query('not Survived.isna()')
    pprint(df.info())

    df = df_onehot_embedding(df)
    pprint(df.info())

    corr = df.corr()
    # pprint(corr)
    pprint(list(corr['Survived_0.0'].items()))
    pprint(list(corr['Survived_1.0'].items()))
    # plot_corr_matrix(df, df.keys(), 3)


def test_param_search():
    data_pack = DatasetPackLoader().load_dataset("titanic")
    train_dataset = data_pack.set['train']
    train_dataset.shuffle()
    train_set, valid_set = train_dataset.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])
    train_Ys = reformat_np_arr(train_Ys, NP_ARRAY_TYPE_INDEX)
    valid_Ys = reformat_np_arr(valid_Ys, NP_ARRAY_TYPE_INDEX)

    from sklearn_like_toolkit.warpper.sklearn_wrapper import skMLP

    path = './temp.pkl'
    # if not os.path.exists(path):
    clf_cls = mlxMLPClf
    # clf_cls = XGBoostClf
    # clf_cls = CatBoostClf
    # clf_cls = skMLP
    # clf_cls = MultiLayerPerceptron
    base_clf = clf_cls()
    base_clf.fit(train_Xs, train_Ys)
    # base_clf.dump(path)

    # base_clf = clf_cls().load(path)
    pprint(base_clf.score(valid_Xs, valid_Ys))
    # pprint(base_clf.score_pack(valid_Xs, valid_Ys))
    # pprint(base_clf.predict_confidence(valid_Xs[:1]))

    base_clf = clf_cls()
    param_search = wrapper_GridSearchCV(base_clf, clf_cls.tuning_grid)

    param_search.fit(train_Xs, train_Ys)
    result = param_search.cv_results_
    pprint(result)

    # best_clf = param_search.best_estimator_
    # pprint(best_clf)
    # score = best_clf.score(valid_Xs, valid_Ys)
    # pprint(score)
    #
    # score = best_clf.score_pack(valid_Xs, valid_Ys)
    # pprint(score)

    pass


from sklearn.model_selection import GridSearchCV as _GridSearchCV


class wrapper_GridSearchCV(_GridSearchCV):

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score="warn"):
        super().__init__(estimator, param_grid, scoring, fit_params, n_jobs, iid, refit, cv, verbose, pre_dispatch,
                         error_score, return_train_score)

    def fit(self, X, y=None, groups=None, **fit_params):
        return super().fit(X, y, groups, **fit_params)

    def score(self, X, y=None):
        return super().score(X, y)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def predict_log_proba(self, X):
        return super().predict_log_proba(X)


@deco_timeit
def main():
    class meta_A(type):

        def __init__(cls, name, bases, cls_dict):
            type.__init__(cls, name, bases, cls_dict)
            print('meta_A')

    class meta_B(type):

        def __init__(cls, name, bases, cls_dict):
            type.__init__(cls, name, bases, cls_dict)
            print('meta_B')

    class A(metaclass=meta_A):
        def __init__(self):
            print('class_A')

    class B(metaclass=meta_B):
        def __init__(self):
            print('class_B')

    class meta_AB(meta_A, meta_B):
        def __init__(cls, name, bases, cls_dict):
            meta_A.__init__(cls, name, bases, cls_dict)
            meta_B.__init__(cls, name, bases, cls_dict)
            print('meta_AB')
        pass

    class C(A, B, metaclass=meta_AB):
        def __init__(self):
            A.__init__(self)
            B.__init__(self)

    C()

    # def __init__(self):
    #     print('class C')

    test_param_search()
    pass
