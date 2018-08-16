from hyperopt import hp
from sklearn.ensemble import BaggingRegressor as _BaggingRegressor

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperReg_with_ABC


class skBaggingReg(_BaggingRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None,
                 verbose=0):
        n_estimators = int(n_estimators)
        _BaggingRegressor.__init__(
            self, base_estimator, n_estimators, max_samples, max_features, bootstrap, bootstrap_features,
            oob_score, warm_start, n_jobs, random_state, verbose)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'n_estimators': hp.qloguniform('n_estimators', 2, 5, 1),
    }
    tuning_grid = {
        'base_estimator': None,
        'n_estimators': 10,
        'max_samples': 1.0,
        'max_features': 1.0,
        'bootstrap': True,
        'bootstrap_features': False,
        'oob_score': False,
        'warm_start': False,
        'n_jobs': 1,
        'random_state': None,
        'verbose': 0,
    }