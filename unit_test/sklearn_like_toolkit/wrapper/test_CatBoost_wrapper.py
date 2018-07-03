from script.sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostReg
from unit_test.sklearn_like_toolkit.wrapper.helper import wrapper_reg_common


def test_CatoostReg_wrapper():
    reg = CatBoostReg()
    wrapper_reg_common(reg)
