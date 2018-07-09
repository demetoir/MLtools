from sklearn.linear_model import ElasticNetCV as _ElasticNetCVReg

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skElasticNetCvReg(_ElasticNetCVReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True, normalize=False,
                 precompute='auto', max_iter=1000, tol=1e-4, cv=None, copy_X=True, verbose=0, n_jobs=1,
                 positive=False, random_state=None, selection='cyclic'):
        _ElasticNetCVReg.__init__(
            self, l1_ratio, eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, cv,
            copy_X, verbose, n_jobs, positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

    tuning_grid = {
        'l1_ratio': 0.5,
        'eps': 1e-3,
        'n_alphas': 100,
        'alphas': None,
        'fit_intercept': True,
        'normalize': False,
        'precompute': 'auto',
        'max_iter': 1000,
        'tol': 1e-4,
        'cv': None,
        'copy_X': True,
        'verbose': 0,
        'n_jobs': 1,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }