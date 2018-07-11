# -*- coding:utf-8 -*-
import inspect
import numpy as np
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.data_handler.HousePrices import load_merge_set
from script.data_handler.HousePrices_null_handler import HousePrices_null_handler
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.sklearn_like_toolkit.HyperOpt.HyperOpt import HyperOpt
# print(built-in function) is not good for logging
from script.sklearn_like_toolkit.RegressionPack import RegressionPack
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit
from unit_test.sklearn_like_toolkit.test_RegressionPack import get_reg_data
from unit_test.util.test_PlotTools import test_plt_dist

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
from pandas import DataFrame as DF


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


def get_current_func_name():
    return inspect.stack()[0][3]


# def titanic_submit():
#     datapack = DatasetPackLoader().load_dataset('titanic')
#     # datapack = DatasetPackLoader().load_dataset('titanic')
#     train_set = datapack['train']
#     test_set = datapack['test']
#     train_set.shuffle()
#     train, valid = train_set.split()
#
#     train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
#     valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])
#
#     path = './clf_pack.clf'
#     if not os.path.exists(path):
#         train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
#         clf_pack = ClassifierPack()
#         clf_pack.gridSearchCV(train_Xs, train_Ys, cv=10)
#         clf_pack.dump(path)
#
#     clf_pack = ClassifierPack().load(path)
#     # pprint(clf_pack.optimize_result)
#     clf_pack.drop('skQDA')
#     clf_pack.drop('skGaussian_NB')
#     clf_pack.drop('mlxSoftmaxRegressionClf')
#     clf_pack.drop('mlxPerceptronClf')
#     clf_pack.drop('mlxMLP')
#     clf_pack.drop('mlxLogisticRegression')
#     clf_pack.drop('mlxAdaline')
#     clf_pack.drop('skLinear_SVC')
#
#     train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
#     valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])
#     score = clf_pack.score(train_Xs, train_Ys)
#     pprint(score)
#
#     score = clf_pack.score(valid_Xs, valid_Ys)
#     pprint(score)
#     #
#     esm_pack = clf_pack.to_ensembleClfpack()
#     train, valid = train_set.split((2, 7))
#     train_Xs, train_Ys = train.full_batch(['Xs', 'Ys'])
#     valid_Xs, valid_Ys = valid.full_batch(['Xs', 'Ys'])
#
#     esm_pack.fit(train_Xs, train_Ys)
#     pprint(esm_pack.score_pack(train_Xs, train_Ys))
#     pprint(esm_pack.score_pack(valid_Xs, valid_Ys))
#
#     test_Xs = test_set.full_batch(['Xs'])
#
#     predict = esm_pack.predict(test_Xs)['FoldingHardVote']
#     # predict = clf_pack.predict(test_Xs)['skBagging']
#     pprint(predict)
#     pprint(predict.shape)
#     submit_path = './submit.csv'
#     datapack.to_kaggle_submit_csv(submit_path, predict)
#
#     # clf_pack.dump(path)


# def expon10(low, high, base=10):
#     return base ** np.random.uniform(low, high)
#
#
# def test_expon_min_max():
#     low = -4
#     high = 3
#     base = 10
#     ret = []
#     for i in range(10000):
#         ret += [expon10(low, high, base=base)]
#
#     ret.sort()
#     pprint(ret)
#     ret = np.array(ret)
#     ret = np.log10(ret)
#     plt = PlotTools()
#     plt.dist(ret)


def test_param():
    data_pack = DatasetPackLoader().load_dataset('titanic')
    train_Xs, train_Ys, valid_Xs, valid_Ys = get_reg_data()
    clf_name = 'XGBoostClf'

    def fit_clf(params):
        pprint(params)
        train_set = data_pack['train']

        clf_pack = ClassifierPack()
        clf = clf_pack[clf_name]

        cv = 3
        scores = []
        for _ in range(cv):
            train_set.shuffle()
            train_set, valid_set = train_set.split()
            train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
            valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

            clf = clf.__class__(**params)
            clf.fit(train_Xs, train_Ys)
            score = clf.score(valid_Xs, valid_Ys)

            scores += [score]

        return -np.mean(scores)

    def fit_reg(params):
        pprint(params)

        clf_pack = RegressionPack()
        clf = clf_pack[clf_name]

        cv = 3
        scores = []
        for _ in range(cv):
            clf = clf.__class__(**params)
            clf.fit(train_Xs, train_Ys)
            score = clf.score(valid_Xs, valid_Ys)

            scores += [score]

        return np.mean(scores)

    clf_pack = ClassifierPack()
    clf = clf_pack[clf_name]
    space = clf.HyperOpt_space

    opt = HyperOpt()
    best = opt.fit_serial(fit_clf, space, 10)

    # pprint(opt.trials)
    pprint(opt.losses)
    pprint(opt.result)
    pprint(opt.opt_info)
    pprint(opt.best_param)
    pprint(opt.best_loss)


def test_HousePrices():
    dataset_pack = DatasetPackLoader().load_dataset('HousePrices')

    pass


def print_null_col_info(df, key, Y_key):
    col = df[[key]]
    series = df[key]
    print()
    pprint(f'column : "{key}", {series.isna().sum()}/{len(col)}(null/total)')
    pprint(col.describe())
    print('value_counts')
    pprint(series.value_counts())
    groupby = df[[key, Y_key]].groupby(key).agg(['mean', 'std', 'min', 'max', 'count'])
    pprint(groupby)
    print()


df_keys = [
    '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',
    'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
    'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
    'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',
    'Exterior2nd', 'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation',
    'FullBath', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond',
    'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'GrLivArea',
    'HalfBath', 'Heating', 'HeatingQC', 'HouseStyle', 'Id', 'KitchenAbvGr',
    'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig',
    'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning',
    'MasVnrArea', 'MasVnrType', 'MiscFeature', 'MiscVal', 'MoSold',
    'Neighborhood', 'OpenPorchSF', 'OverallCond', 'OverallQual',
    'PavedDrive', 'PoolArea', 'PoolQC', 'RoofMatl', 'RoofStyle',
    'SaleCondition', 'SalePrice', 'SaleType', 'ScreenPorch', 'Street',
    'TotRmsAbvGrd', 'TotalBsmtSF', 'Utilities', 'WoodDeckSF', 'YearBuilt',
    'YearRemodAdd', 'YrSold'
]
df_Xs_keys = [
    '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',
    'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
    'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
    'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',
    'Exterior2nd', 'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation',
    'FullBath', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond',
    'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'GrLivArea',
    'HalfBath', 'Heating', 'HeatingQC', 'HouseStyle', 'Id', 'KitchenAbvGr',
    'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig',
    'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning',
    'MasVnrArea', 'MasVnrType', 'MiscFeature', 'MiscVal', 'MoSold',
    'Neighborhood', 'OpenPorchSF', 'OverallCond', 'OverallQual',
    'PavedDrive', 'PoolArea', 'PoolQC', 'RoofMatl', 'RoofStyle',
    'SaleCondition', 'SaleType', 'ScreenPorch', 'Street', 'TotRmsAbvGrd',
    'TotalBsmtSF', 'Utilities', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd',
    'YrSold'
]
df_Ys_key = 'SalePrice'


def is_categorical(df, key):
    col = df[[key]]
    series = df[key]
    value_counts = series.value_counts()
    # pprint(value_counts)
    pprint(key, len(value_counts))

    pass


def test_null_handling():
    def df_column_has_null(df: DF) -> DF:
        null_column = df.columns[df.isna().any()].tolist()
        return df.loc[:, null_column]

    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\HousePrices"""
    merge_df = load_merge_set(dataset_path)
    pprint(merge_df.info())
    pprint(df_column_has_null(merge_df).info())

    null_column_df = df_column_has_null(merge_df)

    null_handler = HousePrices_null_handler(null_column_df, df_Xs_keys, 'col_70_SalePrice', silent=False)
    null_handler.boilerplate_maker(null_column_df, path='./gen_code.py')
    # null_column_df = null_handler.execute()
    pprint(df_column_has_null(null_column_df).info())
    null_handler.gen_info()


    # for key in null_column_df.keys():
    #     is_categorical(null_column_df, key)


@deco_timeit
def main():
    test_null_handling()

    # test_HousePrices()

    # test_regpack_HyperOpt_serial()
    # test_regpack_HyperOpt_parallel()
    # test_wrapperclfpack_HyperOpt_serial()
    # test_wrapperclfpack_HyperOpt_parallel()
    pass
