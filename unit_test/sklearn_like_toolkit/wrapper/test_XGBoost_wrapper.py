from script.sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostReg
from unit_test.sklearn_like_toolkit.wrapper.helper import helper_wrapper_reg_common


def test_XGBoostReg_wrapper():
    reg = XGBoostReg()
    helper_wrapper_reg_common(reg)
