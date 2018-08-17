from hyperopt import hp
from sklearn.linear_model import Ridge as _RidgeReg

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skRidgeReg(_RidgeReg, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-3,
                 solver="auto", random_state=None):
        _RidgeReg.__init__(self, alpha, fit_intercept, normalize, copy_X, max_iter, tol, solver, random_state)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'alpha': hp.loguniform('alpha', -5, 3),
        'max_iter': hp.qloguniform('max_iter', 4, 8, 1),
        'tol': hp.loguniform('tol', -8, 0),

    }
    tuning_grid = {
        'alpha': 1.0,
        'tol': 1e-3,

        # 'fit_intercept': True,
        # 'normalize': False,
        # 'max_iter': None,

        # 'copy_X': True,
        # 'solver': "auto",
        # 'random_state': None,
    }
