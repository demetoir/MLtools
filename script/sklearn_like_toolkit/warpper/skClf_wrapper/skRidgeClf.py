from hyperopt import hp
from sklearn.linear_model import RidgeClassifier as _RidgeClassifier

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperClf_with_ABC


class skRidgeClf(_RidgeClassifier, BaseWrapperClf, metaclass=MetaBaseWrapperClf_with_ABC):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-3,
                 class_weight=None, solver="auto", random_state=None):
        _RidgeClassifier.__init__(
            self, alpha, fit_intercept, normalize, copy_X, max_iter, tol, class_weight, solver, random_state)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'alpha': hp.loguniform('alpha', -5, 3),
        'tol': 1e-5,
        'max_iter': 1000,
    }
    tuning_grid = {
        #  positive
        'alpha': [1.0],

        'tol': 1e-3,

        # 'fit_intercept': True,
        # 'normalize': False,

        # 'copy_X': True,
        # 'class_weight': None,
        # 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        # 'random_state': None,
    }
