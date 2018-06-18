from sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from sklearn_like_toolkit.base.BaseWrapperPack import BaseWrapperPack
from sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostClf
from sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxAdalineClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxLogisticRegressionClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxMLPClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxPerceptronClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxSoftmaxRegressionClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingClf
from sklearn_like_toolkit.warpper.sklearn_wrapper import skMLP
from sklearn_like_toolkit.warpper.sklearn_wrapper import skLinear_SVC
from sklearn_like_toolkit.warpper.sklearn_wrapper import skSGD
from sklearn_like_toolkit.warpper.sklearn_wrapper import skGaussian_NB
from sklearn_like_toolkit.warpper.sklearn_wrapper import skBernoulli_NB
from sklearn_like_toolkit.warpper.sklearn_wrapper import skMultinomial_NB
from sklearn_like_toolkit.warpper.sklearn_wrapper import skDecisionTree
from sklearn_like_toolkit.warpper.sklearn_wrapper import skRandomForest
from sklearn_like_toolkit.warpper.sklearn_wrapper import skExtraTrees
from sklearn_like_toolkit.warpper.sklearn_wrapper import skAdaBoost
from sklearn_like_toolkit.warpper.sklearn_wrapper import skGradientBoosting
from sklearn_like_toolkit.warpper.sklearn_wrapper import skQDA
from sklearn_like_toolkit.warpper.sklearn_wrapper import skKNeighbors
from sklearn_like_toolkit.warpper.sklearn_wrapper import skGaussianProcess
from sklearn_like_toolkit.warpper.sklearn_wrapper import skRBF_SVM
from sklearn_like_toolkit.warpper.sklearn_wrapper import skBagging
from sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostClf


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

    def make_FoldingHardVote(self):
        clfs = [v for k, v in self.pack.items()]
        return FoldingHardVote(clfs)

    def make_stackingClf(self, meta_clf):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingClf(clfs, meta_clf)

    def make_stackingCVClf(self, meta_clf):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingCVClf(clfs, meta_clf)

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

    def clone_clf(self, key, n=1, param=None):
        if key not in self.pack:
            raise KeyError(f"key '{key}' not exist")
