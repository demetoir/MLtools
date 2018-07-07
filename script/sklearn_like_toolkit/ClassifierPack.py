from script.sklearn_like_toolkit.EnsembleClfPack import EnsembleClfPack
from script.sklearn_like_toolkit.FoldingHardVoteClf import FoldingHardVoteClf
from script.sklearn_like_toolkit.base.BaseWrapperClfPack import BaseWrapperClfPack
from script.sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostClf
from script.sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxAdalineClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxLogisticRegressionClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxMLPClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxPerceptronClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxSoftmaxRegressionClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skMLPClf import skMLPClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRidgeCVClf import skRidgeCVClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRidgeClf import skRidgeClf
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skLinear_SVCClf
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skSGDClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skGaussian_NBClf import skGaussian_NBClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skBernoulli_NBClf import skBernoulli_NBClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skMultinomial_NBClf import skMultinomial_NBClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skDecisionTreeClf import skDecisionTreeClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skRandomForestClf import skRandomForestClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skExtraTreesClf import skExtraTreesClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skAdaBoostClf import skAdaBoostClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skGradientBoostingClf import skGradientBoostingClf
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skQDAClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skKNeighborsClf import skKNeighborsClf
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skGaussianProcessClf
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skRBF_SVMClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skBaggingClf import skBaggingClf
from script.sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostClf


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
        skQDAClf.__name__: skQDAClf,
        skKNeighborsClf.__name__: skKNeighborsClf,
        skLinear_SVCClf.__name__: skLinear_SVCClf,
        skRBF_SVMClf.__name__: skRBF_SVMClf,
        skGaussianProcessClf.__name__: skGaussianProcessClf,
        skBaggingClf.__name__: skBaggingClf,
        XGBoostClf.__name__: XGBoostClf,
        LightGBMClf.__name__: LightGBMClf,
        CatBoostClf.__name__: CatBoostClf,
        mlxAdalineClf.__name__: mlxAdalineClf,
        mlxLogisticRegressionClf.__name__: mlxLogisticRegressionClf,
        mlxMLPClf.__name__: mlxMLPClf,
        mlxPerceptronClf.__name__: mlxPerceptronClf,
        mlxSoftmaxRegressionClf.__name__: mlxSoftmaxRegressionClf,
        skRidgeClf.__name__:skRidgeClf,
        skRidgeCVClf.__name__:skRidgeCVClf
    }

    def to_FoldingHardVote(self):
        clfs = [v for k, v in self.pack.items()]
        return FoldingHardVoteClf(clfs)

    def to_stackingClf(self, meta_clf=None):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingClf(clfs, meta_clf)

    def to_stackingCVClf(self, meta_clf=None):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingCVClf(clfs, meta_clf)

    def to_ensembleClfpack(self, meta_clf=None):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return EnsembleClfPack(clfs, meta_clf)

    def clone_top_k_tuned(self, k=5):
        new_pack = {}
        for key in self.pack:
            new_pack[key] = self.pack[key]
            results = self.optimize_result[key][1:k]

            for i, result in enumerate(results):
                param = result["param"]
                cls = self.pack[key].__class__
                new_key = str(cls.__name__) + '_' + str(i + 1)
                clf = cls(**param)
                new_pack[new_key] = clf

        self.pack = new_pack
        return self.pack

    def add_clf(self, key, clf):
        if key in self.pack:
            raise KeyError(f"key '{key}' is not unique")

        self.pack[key] = clf
