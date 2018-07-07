from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxLinearReg, mlxStackingCVReg, mlxStackingReg
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skBayesianRidgeReg, skDecisionTreeReg, skMLPReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoReg import skLassoReg
from unit_test.sklearn_like_toolkit.wrapper.helper import wrapper_reg_common


def test_mlxLinearReg():
    reg = mlxLinearReg()
    wrapper_reg_common(reg)


def test_mlxStackingCVReg():
    meta = skBayesianRidgeReg()
    regs = [skDecisionTreeReg(), skLassoReg(), skMLPReg()]
    reg = mlxStackingCVReg(regs, meta)
    wrapper_reg_common(reg)


def test_mlxStackingReg():
    meta = skBayesianRidgeReg()
    regs = [skDecisionTreeReg(), skLassoReg(), skMLPReg()]
    reg = mlxStackingReg(regs, meta)
    wrapper_reg_common(reg)
