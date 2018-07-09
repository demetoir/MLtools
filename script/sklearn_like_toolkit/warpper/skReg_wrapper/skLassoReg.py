from hyperopt import hp
from sklearn.linear_model import Lasso as _LassoReg

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skLassoReg(_LassoReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        _LassoReg.__init__(
            self, alpha, fit_intercept, normalize, precompute, copy_X, max_iter, tol, warm_start, positive,
            random_state, selection)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'alpha': hp.loguniform('alpha', -8, 2),
    }

    tuning_grid = {
        'alpha': 1.0,
        'fit_intercept': True,
        'normalize': False,
        'precompute': False,
        'copy_X': True,
        'max_iter': 1000,
        'tol': 1e-4,
        'warm_start': False,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }
