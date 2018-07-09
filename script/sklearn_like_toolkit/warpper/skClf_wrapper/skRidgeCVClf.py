from hyperopt import hp
from sklearn.linear_model import RidgeClassifierCV as _RidgeClassifierCV

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC


class skRidgeCVClf(_RidgeClassifierCV, BaseWrapperClf, metaclass=meta_BaseWrapperClf_with_ABC):

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
