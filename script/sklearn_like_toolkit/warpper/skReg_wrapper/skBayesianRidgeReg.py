from hyperopt import hp
from sklearn.linear_model import BayesianRidge as _BayesianRidgeReg

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skBayesianRidgeReg(_BayesianRidgeReg, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6,
                 compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
        _BayesianRidgeReg.__init__(
            self, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score, fit_intercept, normalize,
            copy_X, verbose)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'n_iter': 10 + hp.randint('n_iter', 500),
        'tol': hp.loguniform('tol', -8, 0),
        'alpha_1': hp.loguniform('alpha_1', -8, 0),
        'alpha_2': hp.loguniform('alpha_2', -8, 0),
        'lambda_1': hp.loguniform('lambda_1', -8, 0),
        'lambda_2': hp.loguniform('lambda_2', -8, 0),
    }

    tuning_grid = {
        'n_iter': 300,
        'tol': 1.e-3,
        'alpha_1': 1.e-6,
        'alpha_2': 1.e-6,
        'lambda_1': 1.e-6,
        'lambda_2': 1.e-6,
        'compute_score': False,
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'verbose': False,
    }
