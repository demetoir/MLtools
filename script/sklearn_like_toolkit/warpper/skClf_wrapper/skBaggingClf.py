from hyperopt import hp
from sklearn.ensemble import BaggingClassifier as _BaggingClassifier

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC


class skBaggingClf(BaseWrapperClf, _BaggingClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0):
        n_estimators = int(n_estimators)
        _BaggingClassifier.__init__(
            self, base_estimator, n_estimators, max_samples, max_features, bootstrap, bootstrap_features, oob_score,
            warm_start, n_jobs, random_state, verbose)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = hp.choice('classifier_type', [
        {
            'n_estimators': hp.qloguniform('n_estimators', 2, 5, 1),
        },
    ])

    tuning_grid = {

    }

    tuning_params = {
    }
