from hyperopt import hp
from sklearn.linear_model import ElasticNet as _ElasticNetReg

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skElasticNetReg(_ElasticNetReg, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False,
                 max_iter=1000, copy_X=True, tol=1e-4, warm_start=False, positive=False, random_state=None,
                 selection='cyclic'):
        _ElasticNetReg.__init__(
            self, alpha, l1_ratio, fit_intercept, normalize, precompute, max_iter, copy_X, tol, warm_start,
            positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'l1_ratio': hp.uniform('l1_ratio', 0, 1),
        'max_iter': hp.qloguniform('max_iter', 5, 8, 1),
        'tol': hp.loguniform('tol', -8, 0),
        'alpha': hp.uniform('alpha', 0, 1),
    }

    tuning_grid = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'normalize': False,
        'precompute': False,
        'max_iter': 1000,
        'copy_X': True,
        'tol': 1e-4,
        'warm_start': False,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }