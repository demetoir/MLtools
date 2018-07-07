from sklearn.linear_model import TheilSenRegressor as _TheilSenRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skTheilSenReg(_TheilSenRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, fit_intercept=True, copy_X=True, max_subpopulation=1e4, n_subsamples=None, max_iter=300,
                 tol=1.e-3, random_state=None, n_jobs=1, verbose=False):
        _TheilSenRegressor.__init__(
            self, fit_intercept, copy_X, max_subpopulation, n_subsamples, max_iter, tol, random_state, n_jobs,
            verbose)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

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