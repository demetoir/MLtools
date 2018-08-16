from hyperopt import hp
from sklearn.linear_model import Lasso as _LassoReg

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skLassoReg(_LassoReg, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        _LassoReg.__init__(
            self, alpha, fit_intercept, normalize, precompute, copy_X, max_iter, tol, warm_start, positive,
            random_state, selection)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'alpha': hp.uniform('alpha', 0, 1),
        'max_iter': hp.qloguniform('max_iter', 4, 8, 1),
        'tol': hp.loguniform('tol', -8, 0),
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
