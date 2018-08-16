import numpy as np
from hyperopt import hp
from sklearn.linear_model import LassoLars as _LassoLars

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperReg_with_ABC


class skLassoLarsReg(_LassoLars, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto',
                 max_iter=500, eps=np.finfo(np.float).eps, copy_X=True, fit_path=True, positive=False):
        _LassoLars.__init__(
            self, alpha, fit_intercept, verbose, normalize, precompute, max_iter, eps, copy_X, fit_path, positive)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'eps': hp.loguniform('eps', -5, 0),
        'max_iter': hp.qloguniform('max_iter', 4, 8, 1),
        'alpha': hp.uniform('alpha', 0, 1),
    }

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
