from script.sklearn_like_toolkit.RegressionPack import RegressionPack
from unit_test.sklearn_like_toolkit.wrapper.helper import wrapper_reg_common


def test_regpack():
    reg_pack = RegressionPack()
    wrapper_reg_common(reg_pack)
