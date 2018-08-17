from script.sklearn_like_toolkit.warpper.base.BaseWrapperRegPack import BaseWrapperRegPack
from script.sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostReg
from script.sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMReg
from script.sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostReg
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingReg, mlxStackingCVReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skHuberReg import skHuberReg
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
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skMLPReg import skMLPReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoCVReg import skLassoCVReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skLassoReg import skLassoReg
from script.sklearn_like_toolkit.warpper.skReg_wrapper.skRidgeReg import skRidgeReg


class RegressionPack(BaseWrapperRegPack):

    class_pack = {
        CatBoostReg.__name__: CatBoostReg,
        LightGBMReg.__name__: LightGBMReg,
        XGBoostReg.__name__: XGBoostReg,
        skRidgeReg.__name__: skRidgeReg,
        skMLPReg.__name__: skMLPReg,
        skGradientBoostingReg.__name__: skGradientBoostingReg,
        skBaggingReg.__name__: skBaggingReg,
        skRandomForestReg.__name__: skRandomForestReg,
        skAdaBoostReg.__name__: skAdaBoostReg,
        skExtraTreeReg.__name__: skExtraTreeReg,
        skGaussianProcessReg.__name__: skGaussianProcessReg,
        skDecisionTreeReg.__name__: skDecisionTreeReg,
        skHuberReg.__name__: skHuberReg,
        skRadiusNeighborsReg.__name__: skRadiusNeighborsReg,
        skKNeighborsReg.__name__: skKNeighborsReg,
        # skPassiveAggressiveReg.__name__: skPassiveAggressiveReg,
        skRANSACReg.__name__: skRANSACReg,
        skKernelRidgeReg.__name__: skKernelRidgeReg,
        skTheilSenReg.__name__: skTheilSenReg,
        skElasticNetCvReg.__name__: skElasticNetCvReg,
        skBayesianRidgeReg.__name__: skBayesianRidgeReg,
        skARDReg.__name__: skARDReg,
        skLogisticReg.__name__: skLogisticReg,
        skSGDReg.__name__: skSGDReg,
        skLassoLarsCVReg.__name__: skLassoLarsCVReg,
        skLassoLarsReg.__name__: skLassoLarsReg,
        skLassoCVReg.__name__: skLassoCVReg,
        skLassoReg.__name__: skLassoReg,
        # mlxLinearReg.__name__: mlxLinearReg,
        skElasticNetReg.__name__: skElasticNetReg,
    }
    def to_stackingReg(self, meta_reg=None):
        regs = [reg for k, reg in self.pack.items() if hasattr(reg, 'get_params')]
        return mlxStackingReg(regs, meta_reg)

    def to_stackingCVReg(self, meta_reg=None):
        regs = [reg for k, reg in self.pack.items() if hasattr(reg, 'get_params')]
        return mlxStackingCVReg(regs, meta_reg)

    def to_ensembleRegPack(self):
        pass
