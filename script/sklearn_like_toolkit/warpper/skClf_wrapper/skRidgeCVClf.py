from sklearn.linear_model import RidgeClassifierCV as _RidgeClassifierCV

from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClfWithABC


class skRidgeCVClf(_RidgeClassifierCV, BaseWrapperClf, metaclass=MetaBaseWrapperClfWithABC):

    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=None,
                 class_weight=None):
        _RidgeClassifierCV.__init__(self, alphas, fit_intercept, normalize, scoring, cv, class_weight)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {}

    tuning_grid = {
        # shape positive float
        'alphas': (0.1, 1.0, 10.0),
        # 'cv': None,
        # 'scoring': None,
        # 'class_weight': None
        # 'fit_intercept': True,
        # 'normalize': False,
    }
