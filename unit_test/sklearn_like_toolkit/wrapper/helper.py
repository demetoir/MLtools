from pprint import pprint

from sklearn.metrics import mean_squared_error, r2_score


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

    predict = reg.predict(test_x)
    pprint(predict)

    score = reg.score(train_x, train_y)
    pprint(score)

    # TODO implement
    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error

    mse = mean_squared_error(test_y, predict)
    pprint("Mean squared error: %.2f" % mse)

    # Explained variance score: 1 is perfect prediction
    score = r2_score(test_y, predict)
    pprint('Variance score: %.2f' % score)
