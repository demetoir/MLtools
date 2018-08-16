from hyperopt import hp
from sklearn.linear_model import PassiveAggressiveClassifier as _PassiveAggressiveClassifier
from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClfWithABC


class skPassiveAggressiveClf(_PassiveAggressiveClassifier, BaseWrapperClf, metaclass=MetaBaseWrapperClfWithABC):

    def __init__(self, C=1.0, fit_intercept=True, max_iter=1000, tol=None, shuffle=True, verbose=0, loss="hinge",
                 n_jobs=1, random_state=None, warm_start=False, class_weight=None, average=False, n_iter=None):
        n_jobs = 4
        _PassiveAggressiveClassifier.__init__(
            self, C, fit_intercept, max_iter, tol, shuffle, verbose, loss, n_jobs, random_state, warm_start,
            class_weight, average, n_iter)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'C': hp.loguniform('C', -5, 3),
        'loss': hp.choice('loss', ["hinge", 'squared_hinge']),
    }

    tuning_grid = {
        'C': 1.0,
        'fit_intercept': True,
        # 'max_iter': None,
        'tol': None,
        'shuffle': True,
        'verbose': 0,
        'loss': ["hinge", 'squared_hinge'],
        # 'n_jobs': 1,
        'random_state': None,
        'warm_start': False,
        'class_weight': None,
        'average': False,
        'n_iter': None,
    }
