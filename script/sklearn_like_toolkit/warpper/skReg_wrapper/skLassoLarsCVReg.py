import numpy as np
from sklearn.linear_model import LassoLarsCV as _LassoLarsCV

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skLassoLarsCVReg(_LassoLarsCV, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto', cv=None,
                 max_n_alphas=1000, n_jobs=1, eps=np.finfo(np.float).eps, copy_X=True, positive=False):
        _LassoLarsCV.__init__(
            self, fit_intercept, verbose, max_iter, normalize, precompute, cv, max_n_alphas, n_jobs, eps, copy_X,
            positive)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

    tuning_grid = {
        'fit_intercept': True,
        'verbose': False,
        'max_iter': 500,
        'normalize': True,
        'precompute': 'auto',
        'cv': None,
        'max_n_alphas': 1000,
        'n_jobs': 1,
        'eps': np.finfo(np.float).eps,
        'copy_X': True,
        'positive': False,
    }