from pprint import pprint

from script.sklearn_like_toolkit.RegressionPack import RegressionPack
from unit_test.sklearn_like_toolkit.wrapper.helper import helper_wrapper_reg_common


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


def test_regpack():
    reg_pack = RegressionPack()
    helper_wrapper_reg_common(reg_pack)


def test_regpack_to_stacking():
    train_Xs, train_Ys, test_Xs, test_Ys = get_reg_data()
    reg_pack = RegressionPack()
    reg_pack.fit(train_Xs, train_Ys)

    stacking = reg_pack.to_stackingReg(meta_reg=reg_pack['skMLPReg'])
    helper_wrapper_reg_common(stacking)


def test_regpack_to_stackingCV():
    train_Xs, train_Ys, test_Xs, test_Ys = get_reg_data()
    reg_pack = RegressionPack()
    reg_pack.fit(train_Xs, train_Ys)

    stackingCV = reg_pack.to_stackingCVReg(meta_reg=reg_pack['skMLPReg'])
    helper_wrapper_reg_common(stackingCV)


def test_regpack_HyperOpt_serial():
    train_Xs, train_Ys, test_Xs, test_Ys = get_reg_data()
    reg_pack = RegressionPack()

    reg_pack.HyperOptSearch(train_Xs, train_Ys, 5)
    pprint(reg_pack.HyperOpt_best_loss)

    score_pack = reg_pack.score_pack(test_Xs, test_Ys)
    pprint(score_pack)


def test_regpack_HyperOpt_parallel():
    train_Xs, train_Ys, test_Xs, test_Ys = get_reg_data()
    reg_pack = RegressionPack()

    reg_pack.HyperOptSearch(train_Xs, train_Ys, 5, parallel=True)
    pprint(reg_pack.HyperOpt_best_loss)

    score_pack = reg_pack.score_pack(test_Xs, test_Ys)
    pprint(score_pack)
