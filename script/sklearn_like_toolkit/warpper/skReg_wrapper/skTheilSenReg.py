from hyperopt import hp
from sklearn.linear_model import TheilSenRegressor as _TheilSenRegressor

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperReg_with_ABC


class skTheilSenReg(_TheilSenRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, fit_intercept=True, copy_X=True, max_subpopulation=1e4, n_subsamples=None, max_iter=300,
                 tol=1.e-3, random_state=None, n_jobs=1, verbose=False):
        max_iter = int(max_iter)
        _TheilSenRegressor.__init__(
            self, fit_intercept, copy_X, max_subpopulation, n_subsamples, max_iter, tol, random_state, n_jobs,
            verbose)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'max_subpopulation': 1e4,
        'n_subsamples': None,
        'max_iter': hp.qloguniform('max_iter', 4, 8, 1),
        'tol': hp.loguniform('tol', -8, 0),
    }

    tuning_grid = {
        'fit_intercept': True,
        'copy_X': True,
        'max_subpopulation': 1e4,
        'n_subsamples': None,
        'max_iter': 300,
        'tol': 1.e-3,
        'random_state': None,
        'n_jobs': 1,
        'verbose': False,
    }