from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxLinearReg, mlxStackingCVReg, mlxStackingReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skBayesianRidgeReg import skBayesianRidgeReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skDecisionTreeReg import skDecisionTreeReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skMLPReg import skMLPReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoReg import skLassoReg
from unit_test.sklearn_like_toolkit.wrapper.helper import helper_wrapper_reg_common


def test_mlxLinearReg():
    reg = mlxLinearReg()
    helper_wrapper_reg_common(reg)


def test_mlxStackingCVReg():
    meta = skBayesianRidgeReg()
    regs = [skDecisionTreeReg(), skLassoReg(), skMLPReg()]
    reg = mlxStackingCVReg(regs, meta)
    helper_wrapper_reg_common(reg)


def test_mlxStackingReg():
    meta = skBayesianRidgeReg()
    regs = [skDecisionTreeReg(), skLassoReg(), skMLPReg()]
    reg = mlxStackingReg(regs, meta)
    helper_wrapper_reg_common(reg)
