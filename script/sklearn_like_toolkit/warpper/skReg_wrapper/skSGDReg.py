from hyperopt import hp
from sklearn.linear_model import SGDRegressor as _SGDRegressor
from sklearn.linear_model.stochastic_gradient import DEFAULT_EPSILON

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skSGDReg(_SGDRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                 max_iter=1000, tol=None, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.25, warm_start=False, average=False, n_iter=None):
        _SGDRegressor.__init__(
            self, loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon,
            random_state, learning_rate, eta0, power_t, warm_start, average, n_iter)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'alpha': hp.loguniform('alpha', -6, 3),
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'optimal']),
        'tol': 0.0001,
        'max_iter': 1000,
        'penalty': hp.choice('penalty', ['none', 'l1', 'l2', 'elasticnet']),
        'loss': hp.choice('loss', ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
        'eta0': hp.loguniform('eta0', -4, 1),
    }

    tuning_grid = {

        'loss': "squared_loss",
        'penalty': "l2",
        'alpha': 0.0001,
        'l1_ratio': 0.15,
        'fit_intercept': True,
        'max_iter': None,
        'tol': None,
        'shuffle': True,
        'verbose': 0,
        'epsilon': DEFAULT_EPSILON,
        'random_state': None,
        'learning_rate': "invscaling",
        'eta0': 0.01,
        'power_t': 0.25,
        'warm_start': False,
        'average': False,
        'n_iter': None,
    }