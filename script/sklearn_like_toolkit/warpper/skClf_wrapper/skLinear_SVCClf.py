from hyperopt import hp
from sklearn.svm import LinearSVC as _skLinearSVC

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperClf_with_ABC


class skLinear_SVCClf(BaseWrapperClf, _skLinearSVC, metaclass=MetaBaseWrapperClf_with_ABC):
    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr',
                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                 max_iter=1000):
        _skLinearSVC.__init__(
            self, penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling, class_weight, verbose,
            random_state, max_iter)

        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'C': hp.loguniform('C', -4, 3),
        'loss': hp.choice('loss', ['squared_hinge', 'hinge']),
        # 'penalty': hp.choice('penalty', ['l2', 'l1']),
    }

    tuning_grid = {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        'max_iter': [2 ** i for i in range(6, 13)],
    }
    only_default_params = {
        'fit_intercept': True,
        'intercept_scaling': 1,

        # todo ???
        'multi_class': ['ovr', 'crammer_singer'],
        'loss': ['squared_hinge', 'hinge'],
        'penalty': ['l2', 'l1'],
        'class_weight': None,
        'dual': True,

    }
    etc_param = {
        'random_state': None,
        'tol': 0.0001,
        'verbose': 1e-4,
    }
