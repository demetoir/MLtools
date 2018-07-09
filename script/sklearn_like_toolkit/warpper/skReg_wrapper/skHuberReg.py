from sklearn.isotonic import IsotonicRegression as _IsotonicRegression
from sklearn.linear_model import HuberRegressor as _HuberRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skHuberReg(_HuberRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
        _HuberRegressor.__init__(self, epsilon, max_iter, alpha, warm_start, fit_intercept, tol)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

    tuning_grid = {
        'epsilon': 1.35,
        'max_iter': 100,
        'alpha': 0.0001,
        'warm_start': False,
        'fit_intercept': True,
        'tol': 1e-05,
    }


class skIsotonicReg(_IsotonicRegression, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, y_min=None, y_max=None, increasing=True, out_of_bounds='nan'):
        _IsotonicRegression.__init__(self, y_min, y_max, increasing, out_of_bounds)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

    tuning_grid = {
        'y_min': None,
        'y_max': None,
        'increasing': True,
        'out_of_bounds': 'nan',
    }