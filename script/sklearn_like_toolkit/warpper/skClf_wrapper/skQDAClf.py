import warnings

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _skQDA

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf


class skQDAClf(BaseWrapperClf, _skQDA, metaclass=meta_BaseWrapperClf):
    def __init__(self, priors=None, reg_param=0., store_covariance=False, tol=1.0e-4, store_covariances=None):
        warnings.filterwarnings(module='sklearn*', action='ignore', category=Warning)

        BaseWrapperClf.__init__(self)
        _skQDA.__init__(self, priors, reg_param, store_covariance, tol, store_covariances)

    tuning_grid = {
    }
    remain_param = {
        # TODO
        # ? ..
        'priors': None,
        'reg_param': 0.0,
        'store_covariance': False,
        'store_covariances': None,
        'tol': 0.0001
    }