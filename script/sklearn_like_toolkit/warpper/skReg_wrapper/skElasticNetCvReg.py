from hyperopt import hp
from sklearn.linear_model import ElasticNetCV as _ElasticNetCVReg

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperReg_with_ABC


class skElasticNetCvReg(_ElasticNetCVReg, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):
    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True, normalize=False,
                 precompute='auto', max_iter=1000, tol=1e-4, cv=None, copy_X=True, verbose=0, n_jobs=1,
                 positive=False, random_state=None, selection='cyclic'):
        n_alphas = int(n_alphas)
        max_iter = int(max_iter)
        _ElasticNetCVReg.__init__(
            self, l1_ratio, eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, cv,
            copy_X, verbose, n_jobs, positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'l1_ratio': hp.uniform('l1_ratio', 0, 1),
        'eps': hp.loguniform('eps', -7, 0),
        'n_alphas': hp.qloguniform('n_alphas', 4, 8, 1),
        'max_iter': hp.qloguniform('max_iter', 5, 8, 1),
        'tol': hp.loguniform('tol', -8, 0),
    }

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
