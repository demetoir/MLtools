from hyperopt import hp
from sklearn.ensemble import GradientBoostingClassifier as _skGradientBoostingClassifier

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC
import numpy as np


class skGradientBoostingClf(BaseWrapperClf, _skGradientBoostingClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3,
                 min_impurity_decrease=0., min_impurity_split=None, init=None, random_state=None, max_features=None,
                 verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'):
        n_estimators = int(n_estimators)
        _skGradientBoostingClassifier.__init__(
            self, loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_depth, min_impurity_decrease, min_impurity_split, init, random_state,
            max_features, verbose, max_leaf_nodes, warm_start, presort)
        BaseWrapperClf.__init__(self)

    def _make_estimator(self, append=True):
        pass

    HyperOpt_space = {
        'learning_rate': hp.loguniform('learning_rate', -8, 1),
        'n_estimators': hp.qloguniform('n_estimators', 2, 5, 1),
        'max_depth': 1 + hp.randint('max_depth', 10),
        'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
        'min_samples_split': 2 + hp.randint('min_samples_split', 10),
        'loss': hp.choice('loss', ['deviance', 'exponential']),

        # 'max_leaf_nodes': None,
        # 'max_features': None,
        # 'min_impurity_decrease': 0.0,
        # 'class_weight': None,
        # 'min_weight_fraction_leaf': 0.0,
    }
    tuning_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_depth': [i for i in range(1, 10, 2)],
        'n_estimators': [16, 32, 64, 128, 256],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    only_default_params = {
        'criterion': 'friedman_mse',
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'max_features': None,
        'max_leaf_nodes': None,
    }
    etc_param = {
        # todo wtf?
        'init': None,
        'loss': ['deviance', 'exponential'],
        'subsample': 1.0,

        # etc
        'min_weight_fraction_leaf': 0.0,
        'presort': 'auto',
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }

    @property
    def feature_importances(self):
        return self.feature_importances_

    @property
    def feature_importances_pack(self):
        return {
            'mean': self.feature_importances,
            'std': np.std([
                tree.feature_importances_ for tree in self.estimators_
            ], axis=0)
        }
