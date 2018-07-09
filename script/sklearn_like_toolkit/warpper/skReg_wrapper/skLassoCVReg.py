from hyperopt import hp
from sklearn.linear_model import LassoCV as _LassoCVReg

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skLassoCVReg(_LassoCVReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, copy_X=True, cv=None, verbose=False, n_jobs=1, positive=False,
                 random_state=None, selection='cyclic'):
        n_alphas = int(n_alphas)
        _LassoCVReg.__init__(
            self, eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, copy_X, cv,
            verbose, n_jobs, positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'n_alphas': hp.qloguniform('n_alphas', 2, 5, 1),
        'eps': hp.loguniform('eps', -5, 0)
    }

    tuning_grid = {
        'eps': 1e-3,
        'n_alphas': 100,
        'alphas': None,
        'fit_intercept': True,
        'normalize': False,
        'precompute': 'auto',
        'max_iter': 1000,
        'tol': 1e-4,
        'copy_X': True,
        'cv': None,
        'verbose': False,
        'n_jobs': 1,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }
