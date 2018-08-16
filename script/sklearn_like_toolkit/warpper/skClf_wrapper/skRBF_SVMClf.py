from hyperopt import hp
from sklearn.svm import SVC as _SVC

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperClf_with_ABC


class skRBF_SVMClf(BaseWrapperClf, _SVC, metaclass=MetaBaseWrapperClf_with_ABC):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                 tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        _SVC.__init__(
            self, C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose,
            max_iter, decision_function_shape, random_state)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'C': hp.loguniform('C', -4, 3),
        'gamma': hp.loguniform('gamma', -4, 3),
    }

    tuning_grid = {
        'C': [1 ** i for i in range(-5, 5)],
        'gamma': [1 ** i for i in range(-5, 5)],
    }
    # todo
    remain_param = {
        'cache_size': 200,
        'class_weight': None,
        'coef0': 0.0,
        'decision_function_shape': 'ovr',
        'degree': 3,
        'kernel': 'rbf',
        'max_iter': -1,
        'probability': False,
        'random_state': None,
        'shrinking': True,
        'tol': 0.001,
        'verbose': False
    }
