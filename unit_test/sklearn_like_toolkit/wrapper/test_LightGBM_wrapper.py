from sklearn.metrics import mean_squared_error, r2_score
from script.sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMReg
from unit_test.sklearn_like_toolkit.wrapper.helper import get_reg_data


def test_LightGBMReg_wrapper():
    train_x, train_y, test_x, test_y = get_reg_data()

    # Create linear regression object
    regr = LightGBMReg()

    # Train the model using the training sets
    regr.fit(train_x, train_y)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(test_x)
    # pprint(diabetes_y_pred)

    score = regr.score(train_x, train_y)
    # pprint(score)

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error

    mse = mean_squared_error(test_y, diabetes_y_pred)
    # print("Mean squared error: %.2f" % mse)

    # Explained variance score: 1 is perfect prediction
    score = r2_score(test_y, diabetes_y_pred)
    # print('Variance score: %.2f' % score)
