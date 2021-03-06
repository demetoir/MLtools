from hyperopt import hp
from sklearn.ensemble import ExtraTreesClassifier as _skExtraTreesClassifier

from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClfWithABC
import numpy as np


class skExtraTreesClf(BaseWrapperClf, _skExtraTreesClassifier, metaclass=MetaBaseWrapperClfWithABC):

    def __init__(self, n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 warm_start=False, class_weight=None):
        n_estimators = int(n_estimators)
        _skExtraTreesClassifier.__init__(
            self, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start, class_weight)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'n_estimators': hp.qloguniform('n_estimators', 2, 5, 1),
        'max_depth': 1 + hp.randint('max_depth', 10),
        'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
        'min_samples_split': 2 + hp.randint('min_samples_split', 10),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),

        # 'max_leaf_nodes': None,
        # 'max_features': None,
        # 'min_impurity_decrease': 0.0,
        # 'class_weight': None,
        # 'min_weight_fraction_leaf': 0.0,
    }

    tuning_grid = {
        'n_estimators': [16, 32, 64],
        'max_depth': [i for i in range(1, 10, 2)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    only_default_params = {
        'bootstrap': False,
        'criterion': 'gini',
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
    }
    etc_param = {
        # class weight options
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'n_jobs': 1,
        'oob_score': False,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }

    @property
    def feature_importances(self):
        return np.mean([
            tree.feature_importances_ for tree in self.estimators_
        ], axis=0)

    @property
    def feature_importances_pack(self):
        return {
            'mean': self.feature_importances,
            'std': np.std([
                tree.feature_importances_ for tree in self.estimators_
            ], axis=0)
        }
