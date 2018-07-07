from hyperopt import hp
from sklearn.linear_model import SGDClassifier as _skSGDClassifier
from sklearn.linear_model.stochastic_gradient import DEFAULT_EPSILON

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC


class skSGDClf(BaseWrapperClf, _skSGDClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    # todo wtf?
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                 tol=None, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=1, random_state=None,
                 learning_rate="optimal", eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False,
                 n_iter=None):
        _skSGDClassifier.__init__(
            self, loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon, n_jobs,
            random_state, learning_rate, eta0, power_t, class_weight, warm_start, average, n_iter)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = hp.choice('classifier_type', [{
        'alpha': hp.loguniform('alpha', -6, 3),
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'optimal']),
        'tol': 0.0001,
        'max_iter': 1000,
        'penalty': hp.choice('penalty', ['none', 'l1', 'l2', 'elasticnet']),
        'loss': hp.choice('loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
        'eta0': hp.loguniform('eta0', -4, 1),
    }
    ])
    tuning_grid = {
        # todo random..
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 1.0],
        'tol': 0.0001,
        'max_iter': 1000,
        'learning_rate': ['optimal', 'constant', 'invscaling'],
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],

    }

    remain_param = {
        # TODO
        'tol': 0.0001,

        # 'alpha': 0.0001,

        'average': False,
        'class_weight': None,
        'epsilon': 0.1,
        'eta0': 0.0,
        'fit_intercept': True,
        'l1_ratio': 0.15,
        'loss': 'hinge',
        'n_iter': None,

        'power_t': 0.5,

        # etc
        'n_jobs': 1,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'shuffle': True,
    }
