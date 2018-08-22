from script.sklearn_like_toolkit.EnsembleClfPack import EnsembleClfPack
from script.sklearn_like_toolkit.FoldingHardVoteClf import FoldingHardVoteClf
from script.sklearn_like_toolkit.warpper.base.BaseWrapperClfPack import BaseWrapperClfPack
from script.sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostClf
from script.sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skAdaBoostClf import skAdaBoostClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skBaggingClf import skBaggingClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skBernoulli_NBClf import skBernoulli_NBClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skDecisionTreeClf import skDecisionTreeClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skExtraTreesClf import skExtraTreesClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skGaussianProcessClf import skGaussianProcessClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skGaussian_NBClf import skGaussian_NBClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skGradientBoostingClf import skGradientBoostingClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skKNeighborsClf import skKNeighborsClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skLinear_SVCClf import skLinear_SVCClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skMLPClf import skMLPClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skMultinomial_NBClf import skMultinomial_NBClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skPassiveAggressiveClf import skPassiveAggressiveClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRBF_SVMClf import skRBF_SVMClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRandomForestClf import skRandomForestClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRidgeCVClf import skRidgeCVClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skSGDClf import skSGDClf
from script.sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostClf


# TODO
# kernel
# RBF
# martern
# RationalQuadratic
# Dotproduct
# ExpSineSquared


class ClassifierPack(BaseWrapperClfPack):
    class_pack = {
        skMLPClf.__name__: skMLPClf,
        skSGDClf.__name__: skSGDClf,
        skGaussian_NBClf.__name__: skGaussian_NBClf,
        skBernoulli_NBClf.__name__: skBernoulli_NBClf,
        skMultinomial_NBClf.__name__: skMultinomial_NBClf,
        skDecisionTreeClf.__name__: skDecisionTreeClf,
        skRandomForestClf.__name__: skRandomForestClf,
        skExtraTreesClf.__name__: skExtraTreesClf,
        skAdaBoostClf.__name__: skAdaBoostClf,
        skGradientBoostingClf.__name__: skGradientBoostingClf,
        skKNeighborsClf.__name__: skKNeighborsClf,
        skLinear_SVCClf.__name__: skLinear_SVCClf,
        skRBF_SVMClf.__name__: skRBF_SVMClf,
        skGaussianProcessClf.__name__: skGaussianProcessClf,
        skBaggingClf.__name__: skBaggingClf,
        XGBoostClf.__name__: XGBoostClf,
        LightGBMClf.__name__: LightGBMClf,
        CatBoostClf.__name__: CatBoostClf,
        skPassiveAggressiveClf.__name__: skPassiveAggressiveClf,
        skRidgeCVClf.__name__: skRidgeCVClf,
    }

    def to_FoldingHardVote(self):
        return FoldingHardVoteClf([v for k, v in self.pack.items()])

    def to_stackingClf(self, meta_clf=None):
        return mlxStackingClf(
            [
                clf
                for k, clf in self.pack.items()
                if hasattr(clf, 'get_params')
            ],
            meta_clf)

    def to_stackingCVClf(self, meta_clf=None):
        return mlxStackingCVClf(
            [
                clf
                for k, clf in self.pack.items()
                if hasattr(clf, 'get_params')
            ],
            meta_clf)

    def to_ensembleClfpack(self, meta_clf=None):
        return EnsembleClfPack(
            [
                clf
                for k, clf in self.pack.items()
                if hasattr(clf, 'get_params')
            ],
            meta_clf
        )

    def add_clf(self, key, clf):
        if key in self.pack:
            raise KeyError(f"key '{key}' is not unique")

        self.pack[key] = clf
