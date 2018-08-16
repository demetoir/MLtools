from sklearn.linear_model import RidgeCV as _RidgeCVReg

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skRidgeCVReg(_RidgeCVReg, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):
    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=None,
                 gcv_mode=None, store_cv_values=False):
        _RidgeCVReg.__init__(self, alphas, fit_intercept, normalize, scoring, cv, gcv_mode, store_cv_values)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {}

    tuning_grid = {
        # shape positive float
        'alphas': (0.1, 1.0, 10.0),
        # 'fit_intercept': True,
        # 'normalize': False,
        # 'scoring': None,
        # 'cv': None,
        # 'class_weight': None
    }
