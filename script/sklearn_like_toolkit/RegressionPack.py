from script.sklearn_like_toolkit.base.BaseWrapperRegPack import BaseWrapperRegPack
from script.sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostReg
from script.sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMReg
from script.sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostReg
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxLinearReg, mlxStackingReg, mlxStackingCVReg
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skMLPReg, skGradientBoostingReg, \
    skBaggingReg, skRandomForestReg, skAdaBoostReg, skExtraTreeReg, skGaussianProcessReg, skDecisionTreeReg, skHuberReg, \
    skRadiusNeighborsReg, skKNeighborsReg, skPassiveAggressiveReg, skRANSACReg, skKernelRidgeReg, skTheilSenReg, \
    skElasticNetCvReg, skBayesianRidgeReg, skARDReg, skLogisticReg, skSGDReg, skLassoLarsCVReg, skLassoLarsReg, \
    skElasticNetReg
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
        skPassiveAggressiveReg.__name__: skPassiveAggressiveReg,
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
        mlxLinearReg.__name__: mlxLinearReg,
        skElasticNetReg.__name__: skElasticNetReg,
    }

    def to_stackingReg(self, meta_clf=None):
        regs = [reg for k, reg in self.pack.items() if hasattr(reg, 'get_params')]
        return mlxStackingReg(regs, meta_clf)

    def to_stackingCVReg(self, meta_clf=None):
        regs = [reg for k, reg in self.pack.items() if hasattr(reg, 'get_params')]
        return mlxStackingCVReg(regs, meta_clf)

    def to_ensembleRegPack(self):
        pass
