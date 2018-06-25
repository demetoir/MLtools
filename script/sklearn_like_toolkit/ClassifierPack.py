from script.sklearn_like_toolkit.EnsembleClfPack import EnsembleClfPack
from script.sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from script.sklearn_like_toolkit.base.BaseWrapperPack import BaseWrapperPack
from script.sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostClf
from script.sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxAdalineClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxLogisticRegressionClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxMLPClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxPerceptronClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxSoftmaxRegressionClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf
from script.sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingClf
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skMLP
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skLinear_SVC
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skSGD
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skGaussian_NB
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skBernoulli_NB
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skMultinomial_NB
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skDecisionTree
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skRandomForest
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skExtraTrees
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skAdaBoost
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skGradientBoosting
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skQDA
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skKNeighbors
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skGaussianProcess
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skRBF_SVM
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skBagging
from script.sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostClf


class ClassifierPack(BaseWrapperPack):
    class_pack = {
        "skMLP": skMLP,
        "skSGD": skSGD,
        "skGaussian_NB": skGaussian_NB,
        "skBernoulli_NB": skBernoulli_NB,
        "skMultinomial_NB": skMultinomial_NB,
        "skDecisionTree": skDecisionTree,
        "skRandomForest": skRandomForest,
        "skExtraTrees": skExtraTrees,
        "skAdaBoost": skAdaBoost,
        "skGradientBoosting": skGradientBoosting,
        "skQDA": skQDA,
        "skKNeighbors": skKNeighbors,
        "skLinear_SVC": skLinear_SVC,
        "skRBF_SVM": skRBF_SVM,
        "skGaussianProcess": skGaussianProcess,
        "skBagging": skBagging,
        "XGBoost": XGBoostClf,
        "LightGBM": LightGBMClf,
        "CatBoost": CatBoostClf,
        'mlxAdaline': mlxAdalineClf,
        'mlxLogisticRegression': mlxLogisticRegressionClf,
        'mlxMLP': mlxMLPClf,
        'mlxPerceptronClf': mlxPerceptronClf,
        'mlxSoftmaxRegressionClf': mlxSoftmaxRegressionClf,
    }

    def __init__(self, pack_keys=None):
        super().__init__()
        if pack_keys is None:
            pack_keys = self.class_pack.keys()

        self.pack = {}
        for key in pack_keys:
            self.pack[key] = self.class_pack[key]()

    def __getitem__(self, item):
        return self.pack.__getitem__(item)

    def make_FoldingHardVote(self):
        clfs = [v for k, v in self.pack.items()]
        return FoldingHardVote(clfs)

    def make_stackingClf(self, meta_clf=None):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingClf(clfs, meta_clf)

    def make_stackingCVClf(self, meta_clf=None):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingCVClf(clfs, meta_clf)

    def make_ensembleClfpack(self, meta_clf=None):
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

    def drop_clf(self, key):
        self.pack.pop(key)

    def add_clf(self, key, clf):
        if key in self.pack:
            raise KeyError(f"key '{key}' is not unique")

        self.pack[key] = clf
