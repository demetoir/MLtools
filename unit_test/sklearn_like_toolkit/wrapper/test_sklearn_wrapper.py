from script.sklearn_like_toolkit.warpper.skReg_wrapper.skHuberReg import skHuberReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skIsotonicReg import skIsotonicReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skKernelRidgeReg import skKernelRidgeReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skTheilSenReg import skTheilSenReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skRANSACReg import skRANSACReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLogisticReg import skLogisticReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skARDReg import skARDReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skBayesianRidgeReg import skBayesianRidgeReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skElasticNetCvReg import skElasticNetCvReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skElasticNetReg import skElasticNetReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoLarsCVReg import skLassoLarsCVReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoLarsReg import skLassoLarsReg
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
from unit_test.sklearn_like_toolkit.wrapper.helper import helper_wrapper_clf_common, helper_wrapper_reg_common


def test_skRidgeClf():
    clf = skRidgeClf()
    helper_wrapper_clf_common(clf)


def test_skRidgeCVClf():
    clf = skRidgeCVClf()
    helper_wrapper_clf_common(clf)


def test_skPassiveAggressiveClf():
    clf = skPassiveAggressiveClf()
    helper_wrapper_clf_common(clf)


def test_skRadiusNeighborsClf():
    clf = skRadiusNeighborsClf()
    helper_wrapper_clf_common(clf)


def test_skKNeighborsClf():
    clf = skKNeighborsClf()
    helper_wrapper_clf_common(clf)


def test_skNearestCentroidClf():
    clf = skNearestCentroidClf()
    helper_wrapper_clf_common(clf)


def test_skGradientBoostingReg():
    reg = skGradientBoostingReg()
    helper_wrapper_reg_common(reg)


def test_skMLPReg():
    reg = skMLPReg()
    helper_wrapper_reg_common(reg)


def test_skAdaBoostReg():
    reg = skAdaBoostReg()
    helper_wrapper_reg_common(reg)


def test_skRandomForestReg():
    reg = skRandomForestReg()
    helper_wrapper_reg_common(reg)


def test_skBaggingReg():
    reg = skBaggingReg()
    helper_wrapper_reg_common(reg)


def test_skGaussianProcessReg():
    reg = skGaussianProcessReg()
    helper_wrapper_reg_common(reg)


def test_skDecisionTreeReg():
    reg = skDecisionTreeReg()
    helper_wrapper_reg_common(reg)


def test_skExtraTreeReg():
    reg = skExtraTreeReg()
    helper_wrapper_reg_common(reg)


def test_skHuberReg():
    reg = skHuberReg()
    helper_wrapper_reg_common(reg)


def test_skRadiusNeighborsReg():
    reg = skRadiusNeighborsReg()
    helper_wrapper_reg_common(reg)


def test_skKNeighborsReg():
    reg = skKNeighborsReg()
    helper_wrapper_reg_common(reg)


def test_skPassiveAggressiveReg():
    reg = skPassiveAggressiveReg()
    helper_wrapper_reg_common(reg)


def test_skRANSACReg():
    reg = skRANSACReg()
    helper_wrapper_reg_common(reg)


def test_skTheilSenReg():
    reg = skTheilSenReg()
    helper_wrapper_reg_common(reg)


def test_skKernelRidgeReg():
    reg = skKernelRidgeReg()
    helper_wrapper_reg_common(reg)


def test_skElasticNetCvReg():
    reg = skElasticNetCvReg()
    helper_wrapper_reg_common(reg)


def test_skBayesianRidgeReg():
    reg = skBayesianRidgeReg()
    helper_wrapper_reg_common(reg)


def test_skARDReg():
    reg = skARDReg()
    helper_wrapper_reg_common(reg)


def test_skLogisticReg():
    reg = skLogisticReg()
    helper_wrapper_reg_common(reg)


def test_skSGDReg():
    reg = skSGDReg()
    helper_wrapper_reg_common(reg)


def test_skRidgeReg():
    reg = skRidgeReg()
    helper_wrapper_reg_common(reg)


def test_skLassoReg():
    reg = skLassoReg()
    helper_wrapper_reg_common(reg)


def test_skLassoCVReg():
    reg = skLassoCVReg()
    helper_wrapper_reg_common(reg)


def test_skLassoLarsReg():
    reg = skLassoLarsReg()
    helper_wrapper_reg_common(reg)


def test_skLassoLarsCVReg():
    reg = skLassoLarsCVReg()
    helper_wrapper_reg_common(reg)


def test_skElasticNetReg():
    reg = skElasticNetReg()
    helper_wrapper_reg_common(reg)


def test_skIsotonicReg():
    reg = skIsotonicReg()
    helper_wrapper_reg_common(reg)
