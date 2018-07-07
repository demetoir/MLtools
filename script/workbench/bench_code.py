# -*- coding:utf-8 -*-
import os
import time
import numpy as np
from hyperopt import hp
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit
# print(built-in function) is not good for logging
from script.sklearn_like_toolkit.warpper.wrapperRandomizedSearchCV import wrapperRandomizedSearchCV
from script.sklearn_like_toolkit.RegressionPack import RegressionPack
from script.util.PlotTools import PlotTools

from script.util.misc_util import log_error_trace
from script.sklearn_like_toolkit.HyperOpt import HyperOpt
from unit_test.sklearn_like_toolkit.test_HyperOpt import test_hyperOpt
from unit_test.sklearn_like_toolkit.test_wrapperRandomizedSearchCV import test_wrapperRandomizedSearchCV

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


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


def expon10(low, high, base=10):
    return base ** np.random.uniform(low, high)


def test_expon_min_max():
    low = -4
    high = 3
    base = 10
    ret = []
    for i in range(10000):
        ret += [expon10(low, high, base=base)]

    ret.sort()
    pprint(ret)
    ret = np.array(ret)
    ret = np.log10(ret)
    plt = PlotTools()
    plt.dist(ret)


def test_clfpack_wrapperRandomizedSearchCV():
    pass


def test_regpack_wrapperRandomizedSearchCV():
    pass


def get_reg_data():
    import numpy as np
    from sklearn import datasets

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    X_train = diabetes_X[:-20]
    X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]

    return X_train, y_train, X_test, y_test


def test_param():
    data_pack = DatasetPackLoader().load_dataset('titanic')

    clf_name = 'skRadiusNeighborsReg'

    def fit_clf(kwargs):
        params = kwargs['params']
        data_pack = kwargs['data_pack']
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

    def fit_reg(kwargs):
        train_Xs, train_Ys, valid_Xs, valid_Ys = get_reg_data()

        params = kwargs['params']
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

    clf_pack = RegressionPack()
    clf = clf_pack[clf_name]
    space = clf.HyperOpt_space

    opt = HyperOpt()
    best = opt.fit(fit_reg, data_pack, space, 10)

    # pprint(opt.trials)
    pprint(opt.losses)
    pprint(opt.result)
    pprint(opt.opt_info)
    pprint(opt.best_param)
    pprint(opt.best_loss)


@deco_timeit
def main():
    # test_expon_min_max()

    test_param()

    # test_wrapperRandomizedSearchCV()
