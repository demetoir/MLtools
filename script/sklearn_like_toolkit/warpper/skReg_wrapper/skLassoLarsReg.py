import numpy as np
from sklearn.linear_model import LassoLars as _LassoLars

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skLassoLarsReg(_LassoLars, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto',
                 max_iter=500, eps=np.finfo(np.float).eps, copy_X=True, fit_path=True, positive=False):
        _LassoLars.__init__(
            self, alpha, fit_intercept, verbose, normalize, precompute, max_iter, eps, copy_X, fit_path, positive)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

    tuning_grid = {
        'alpha': 1.0,
        'fit_intercept': True,
        'verbose': False,
        'normalize': True,
        'precompute': 'auto',
        'max_iter': 500,
        'eps': np.finfo(np.float).eps,
        'copy_X': True,
        'fit_path': True,
        'positive': False,
    }