from pprint import pprint
from script.data_handler.DatasetPackLoader import DatasetPackLoader


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


def wrapper_reg_common(reg, ):
    train_x, train_y, test_x, test_y = get_reg_data()

    reg.fit(train_x, train_y)

    try:
        predict = reg.predict(test_x)
        pprint(predict)
    except BaseException as e:
        pprint(f" while predict in {reg}, {e}")

    try:
        score = reg.score(train_x, train_y)
        pprint(score)
    except BaseException as e:
        pprint(f" while score in {reg}, {e}")

    # TODO implement
    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error

    try:
        score = reg.score_pack(train_x, train_y)
        pprint(score)
    except BaseException as e:
        pprint(f" while score_pack in {reg}, {e}")


def wrapper_clf_common(clf):
    datapack = DatasetPackLoader().load_dataset("titanic")
    train_set = datapack['train']
    train_set, valid_set = train_set.split((7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])
    sample_Xs, sample_Ys = valid_Xs[:2], valid_Ys[:2]

    clf.fit(train_Xs, train_Ys)

    try:
        predict = clf.predict(valid_Xs)
        print(f'predict {predict}')
    except BaseException as e:
        print(f'while {clf} predict {e}')

    try:
        proba = clf.predict_proba(valid_Xs)
        print(f'proba {proba}')
    except BaseException as e:
        print(f'while {clf} predict_proba {e}')

    try:
        score = clf.score(valid_Xs, valid_Ys)
        print(f'score {score}')
    except BaseException as e:
        print(f'while {clf} score {e}')

    try:
        score = clf.score_pack(valid_Xs, valid_Ys)
        print(f'score_pack {score}')
    except BaseException as e:
        print(f'while {clf} score_pack {e}')
