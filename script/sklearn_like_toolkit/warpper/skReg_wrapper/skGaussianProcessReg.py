from sklearn.gaussian_process import \
    GaussianProcessRegressor as _GaussianProcessRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skGaussianProcessReg(_GaussianProcessRegressor, BaseWrapperReg,
                           metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        _GaussianProcessRegressor.__init__(
            self, kernel, alpha, optimizer, n_restarts_optimizer, normalize_y,
            copy_X_train, random_state)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {}
    tuning_grid = {
        'kernel': None,
        'alpha': 1e-10,
        'optimizer': "fmin_l_bfgs_b",
        'n_restarts_optimizer': 0,
        'normalize_y': False,
        'copy_X_train': True,
        'random_state': None,
    }
