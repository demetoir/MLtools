from hyperopt import hp
from sklearn.linear_model import HuberRegressor as _HuberRegressor

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skHuberReg(_HuberRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
        _HuberRegressor.__init__(self, epsilon, max_iter, alpha, warm_start, fit_intercept, tol)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'max_iter': hp.qloguniform('max_iter', 3, 6, 1),
        'alpha': hp.loguniform('alpha', -7, 0),
        'tol': hp.loguniform('tol', -7, 0),
    }
    tuning_grid = {
        'epsilon': 1.35,
        'max_iter': 100,
        'alpha': 0.0001,
        'tol': 1e-05,

        'warm_start': False,
        'fit_intercept': True,

    }
