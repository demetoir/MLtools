from hyperopt import hp
from sklearn.linear_model import PassiveAggressiveRegressor as _PassiveAggressiveRegressor
from sklearn.linear_model.stochastic_gradient import DEFAULT_EPSILON

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skPassiveAggressiveReg(_PassiveAggressiveRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, C=1.0, fit_intercept=True, max_iter=1000, tol=None, shuffle=True, verbose=0,
                 loss="epsilon_insensitive", epsilon=DEFAULT_EPSILON, random_state=None, warm_start=False,
                 average=False, n_iter=None):
        _PassiveAggressiveRegressor.__init__(
            self, C, fit_intercept, max_iter, tol, shuffle, verbose, loss, epsilon, random_state, warm_start,
            average, n_iter)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'C': hp.loguniform('C', -5, 3),
        'loss': hp.choice('loss', ["epsilon_insensitive", 'squared_epsilon_insensitive']),
        'max_iter': hp.qloguniform('max_iter', 4, 7, 1),
    }

    tuning_grid = {
        'C': 1.0,
        'fit_intercept': True,
        'max_iter': None,
        'tol': None,
        'shuffle': True,
        'verbose': 0,
        'loss': "epsilon_insensitive",
        'epsilon': DEFAULT_EPSILON,
        'random_state': None,
        'warm_start': False,
        'average': False,
        'n_iter': None,
    }
