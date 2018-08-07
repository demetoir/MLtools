from hyperopt import hp
from sklearn.ensemble import AdaBoostClassifier as _skAdaBoostClassifier

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC
import numpy as np


class skAdaBoostClf(BaseWrapperClf, _skAdaBoostClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1., algorithm='SAMME.R', random_state=None):
        n_estimators = int(n_estimators)
        _skAdaBoostClassifier.__init__(self, base_estimator, n_estimators, learning_rate, algorithm, random_state)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'n_estimators': hp.qloguniform('n_estimators', 2, 6, 1),
        'learning_rate': hp.loguniform('learning_rate', -8, 1),
        'algorithm': hp.choice('algorithm', ['SAMME.R', 'SAMME']),
    }

    tuning_grid = {
        'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'n_estimators': [8, 16, 32, 64, 128, 256],
        # 'base_estimator': None,
    }
    etc_param = {
        # etc
        'random_state': None,

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
