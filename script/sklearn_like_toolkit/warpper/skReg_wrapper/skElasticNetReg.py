from sklearn.linear_model import ElasticNet as _ElasticNetReg

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skElasticNetReg(_ElasticNetReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False,
                 max_iter=1000, copy_X=True, tol=1e-4, warm_start=False, positive=False, random_state=None,
                 selection='cyclic'):
        _ElasticNetReg.__init__(
            self, alpha, l1_ratio, fit_intercept, normalize, precompute, max_iter, copy_X, tol, warm_start,
            positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

    tuning_grid = {
        'ffalpha': 1.0,
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