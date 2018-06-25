from sklearn.model_selection import GridSearchCV as _GridSearchCV
from sklearn_like_toolkit.base.MixIn import ClfWrapperMixIn, meta_BaseWrapperClf_with_ABC
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()


class wrapperGridSearchCV(_GridSearchCV, ClfWrapperMixIn, metaclass=meta_BaseWrapperClf_with_ABC):

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None, n_jobs=CPU_COUNT, iid=True, refit=True,
                 cv=None, verbose=20, pre_dispatch='2*n_jobs', error_score='raise', return_train_score="warn"):
        _GridSearchCV.__init__(self, estimator, param_grid, scoring, fit_params, n_jobs, iid, refit, cv, verbose,
                               pre_dispatch,
                               error_score, return_train_score)
        ClfWrapperMixIn.__init__(self)
