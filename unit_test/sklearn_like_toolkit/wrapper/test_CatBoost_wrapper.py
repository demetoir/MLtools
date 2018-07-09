from script.sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostReg
from unit_test.sklearn_like_toolkit.wrapper.helper import helper_wrapper_reg_common


def test_CatoostReg_wrapper():
    reg = CatBoostReg()
    helper_wrapper_reg_common(reg)
