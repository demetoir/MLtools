from script.sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMReg
from unit_test.sklearn_like_toolkit.wrapper.helper import helper_wrapper_reg_common


def test_LightGBMReg_wrapper():
    reg = LightGBMReg()
    helper_wrapper_reg_common(reg)
