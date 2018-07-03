from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxLinearReg
from unit_test.sklearn_like_toolkit.wrapper.helper import wrapper_reg_common


def test_mlxLinearReg():
    reg = mlxLinearReg()
    wrapper_reg_common(reg)
