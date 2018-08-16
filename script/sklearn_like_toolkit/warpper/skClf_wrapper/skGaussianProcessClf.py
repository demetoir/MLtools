from sklearn.gaussian_process import GaussianProcessClassifier as _skGaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF as _RBF

from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperClf_with_ABC


class skGaussianProcessClf(BaseWrapperClf, _skGaussianProcessClassifier, metaclass=MetaBaseWrapperClf_with_ABC):
    def __init__(self, kernel=None, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, max_iter_predict=100,
                 warm_start=False, copy_X_train=True, random_state=None, multi_class="one_vs_rest", n_jobs=1):
        n_jobs = 4
        _skGaussianProcessClassifier.__init__(
            self, kernel, optimizer, n_restarts_optimizer, max_iter_predict, warm_start, copy_X_train, random_state,
            multi_class, n_jobs)
        BaseWrapperClf.__init__(self, )

    HyperOpt_space = {
    }
    tuning_grid = {
    }
    remain_param = {
        'kernel': 1 ** 2 * _RBF(length_scale=1),
        'kernel__k1': 1 ** 2,
        'kernel__k1__constant_value': 1.0,
        'kernel__k1__constant_value_bounds': (1e-05, 100000.0),
        'kernel__k2': _RBF(length_scale=1),
        'kernel__k2__length_scale': 1.0,
        'kernel__k2__length_scale_bounds': (1e-05, 100000.0),

        'max_iter_predict': 100,

        'multi_class': 'one_vs_rest',
        'n_jobs': 1,
        'n_restarts_optimizer': 0,
        'optimizer': 'fmin_l_bfgs_b',
        'random_state': None,
        'warm_start': False,
        'copy_X_train': True,
    }
