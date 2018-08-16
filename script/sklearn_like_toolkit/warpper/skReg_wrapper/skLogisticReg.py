from hyperopt import hp
from sklearn.linear_model import LogisticRegression as _LogisticRegression

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skLogisticReg(_LogisticRegression, BaseWrapperReg,
                    metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver='liblinear',
                 max_iter=100, multi_class='ovr',
                 verbose=0, warm_start=False, n_jobs=1):
        _LogisticRegression.__init__(
            self, penalty, dual, tol, C, fit_intercept, intercept_scaling,
            class_weight, random_state, solver,
            max_iter, multi_class, verbose, warm_start, n_jobs)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        # 'penalty': hp.choice('penalty', ['l1', 'l2']),
        'tol': hp.loguniform('tol', -8, 0),
        'C': hp.loguniform('C', 0, 5),
        'solver': hp.choice('solver',
                            ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
        'max_iter': hp.qloguniform('max_iter', 4, 7, 1),
        # 'multi_class': hp.choice('multi_class', ['ovr', 'multinomial']),
    }

    tuning_grid = {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 1.0,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'class_weight': None,
        'random_state': None,
        'solver': 'liblinear',
        'max_iter': 100,
        'multi_class': 'ovr',
        'verbose': 0,
        'warm_start': False,
        'n_jobs': 1,
    }
