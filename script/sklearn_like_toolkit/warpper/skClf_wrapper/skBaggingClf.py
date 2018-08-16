from hyperopt import hp
from sklearn.ensemble import BaggingClassifier as _BaggingClassifier

from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClfWithABC
import numpy as np


class skBaggingClf(BaseWrapperClf, _BaggingClassifier, metaclass=MetaBaseWrapperClfWithABC):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0):
        n_estimators = int(n_estimators)
        _BaggingClassifier.__init__(
            self, base_estimator, n_estimators, max_samples, max_features, bootstrap, bootstrap_features, oob_score,
            warm_start, n_jobs, random_state, verbose)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'n_estimators': hp.qloguniform('n_estimators', 2, 5, 1),
    }

    tuning_grid = {

    }

    tuning_params = {
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
