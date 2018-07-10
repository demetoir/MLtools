from hyperopt import hp
from sklearn.linear_model import ARDRegression as _ARDRegression

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skARDReg(_ARDRegression, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6,
                 compute_score=False, threshold_lambda=1.e+4, fit_intercept=True, normalize=False, copy_X=True,
                 verbose=False):
        _ARDRegression.__init__(
            self, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score, threshold_lambda,
            fit_intercept, normalize, copy_X, verbose)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'n_iter': 10 + hp.randint('n_iter', 500),
        'tol': hp.loguniform('tol', -8, 0),
        'alpha_1': hp.loguniform('alpha_1', -8, 0),
        'alpha_2': hp.loguniform('alpha_2', -8, 0),
        'lambda_1': hp.loguniform('lambda_1', -8, 0),
        'lambda_2': hp.loguniform('lambda_2', -8, 0),
        'threshold_lambda': hp.loguniform('threshold_lambda', 1, 7),
    }

    tuning_grid = {
        'n_iter': 300,
        'tol': 1.e-3,
        'alpha_1': 1.e-6,
        'alpha_2': 1.e-6,
        'lambda_1': 1.e-6,
        'lambda_2': 1.e-6,
        'compute_score': False,
        'threshold_lambda': 1.e+4,
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'verbose': False,
    }
