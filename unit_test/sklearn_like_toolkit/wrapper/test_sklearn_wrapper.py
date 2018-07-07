from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skHuberReg, \
    skRANSACReg, skTheilSenReg, skKernelRidgeReg, \
    skElasticNetCvReg, skBayesianRidgeReg, skARDReg, skLogisticReg, skLassoLarsReg, skLassoLarsCVReg, skElasticNetReg, skIsotonicReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skRadiusNeighborsReg import skRadiusNeighborsReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skKNeighborsReg import skKNeighborsReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skGaussianProcessReg import skGaussianProcessReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skDecisionTreeReg import skDecisionTreeReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skExtraTreeReg import skExtraTreeReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skAdaBoostReg import skAdaBoostReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skBaggingReg import skBaggingReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skRandomForestReg import skRandomForestReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skGradientBoostingReg import skGradientBoostingReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skSGDReg import skSGDReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skPassiveAggressiveReg import skPassiveAggressiveReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skMLPReg import skMLPReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoCVReg import skLassoCVReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoReg import skLassoReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skRidgeReg import skRidgeReg
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skNearestCentroidClf import skNearestCentroidClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRadiusNeighborsClf import skRadiusNeighborsClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skPassiveAggressiveClf import skPassiveAggressiveClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRidgeCVClf import skRidgeCVClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRidgeClf import skRidgeClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skKNeighborsClf import skKNeighborsClf
from unit_test.sklearn_like_toolkit.wrapper.helper import wrapper_clf_common, wrapper_reg_common


def test_skRidgeClf():
    clf = skRidgeClf()
    wrapper_clf_common(clf)


def test_skRidgeCVClf():
    clf = skRidgeCVClf()
    wrapper_clf_common(clf)


def test_skPassiveAggressiveClf():
    clf = skPassiveAggressiveClf()
    wrapper_clf_common(clf)


def test_skRadiusNeighborsClf():
    clf = skRadiusNeighborsClf()
    wrapper_clf_common(clf)


def test_skKNeighborsClf():
    clf = skKNeighborsClf()
    wrapper_clf_common(clf)


def test_skNearestCentroidClf():
    clf = skNearestCentroidClf()
    wrapper_clf_common(clf)


def test_skGradientBoostingReg():
    reg = skGradientBoostingReg()
    wrapper_reg_common(reg)


def test_skMLPReg():
    reg = skMLPReg()
    wrapper_reg_common(reg)


def test_skAdaBoostReg():
    reg = skAdaBoostReg()
    wrapper_reg_common(reg)


def test_skRandomForestReg():
    reg = skRandomForestReg()
    wrapper_reg_common(reg)


def test_skBaggingReg():
    reg = skBaggingReg()
    wrapper_reg_common(reg)


def test_skGaussianProcessReg():
    reg = skGaussianProcessReg()
    wrapper_reg_common(reg)


def test_skDecisionTreeReg():
    reg = skDecisionTreeReg()
    wrapper_reg_common(reg)


def test_skExtraTreeReg():
    reg = skExtraTreeReg()
    wrapper_reg_common(reg)


def test_skHuberReg():
    reg = skHuberReg()
    wrapper_reg_common(reg)


def test_skRadiusNeighborsReg():
    reg = skRadiusNeighborsReg()
    wrapper_reg_common(reg)


def test_skKNeighborsReg():
    reg = skKNeighborsReg()
    wrapper_reg_common(reg)


def test_skPassiveAggressiveReg():
    reg = skPassiveAggressiveReg()
    wrapper_reg_common(reg)


def test_skRANSACReg():
    reg = skRANSACReg()
    wrapper_reg_common(reg)


def test_skTheilSenReg():
    reg = skTheilSenReg()
    wrapper_reg_common(reg)


def test_skKernelRidgeReg():
    reg = skKernelRidgeReg()
    wrapper_reg_common(reg)


def test_skElasticNetCvReg():
    reg = skElasticNetCvReg()
    wrapper_reg_common(reg)


def test_skBayesianRidgeReg():
    reg = skBayesianRidgeReg()
    wrapper_reg_common(reg)


def test_skARDReg():
    reg = skARDReg()
    wrapper_reg_common(reg)


def test_skLogisticReg():
    reg = skLogisticReg()
    wrapper_reg_common(reg)


def test_skSGDReg():
    reg = skSGDReg()
    wrapper_reg_common(reg)


def test_skRidgeReg():
    reg = skRidgeReg()
    wrapper_reg_common(reg)


def test_skLassoReg():
    reg = skLassoReg()
    wrapper_reg_common(reg)


def test_skLassoCVReg():
    reg = skLassoCVReg()
    wrapper_reg_common(reg)


def test_skLassoLarsReg():
    reg = skLassoLarsReg()
    wrapper_reg_common(reg)


def test_skLassoLarsCVReg():
    reg = skLassoLarsCVReg()
    wrapper_reg_common(reg)


def test_skElasticNetReg():
    reg = skElasticNetReg()
    wrapper_reg_common(reg)


def test_skIsotonicReg():
    reg = skIsotonicReg()
    wrapper_reg_common(reg)
