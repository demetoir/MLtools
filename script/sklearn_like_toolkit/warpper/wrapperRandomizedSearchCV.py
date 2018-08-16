from progressbar import ProgressBar
from sklearn import model_selection
from sklearn.externals.joblib import Parallel
from script.sklearn_like_toolkit.warpper.base.MixIn import ClfWrapperMixIn, MetaBaseWrapperClf_with_ABC
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()


# TODO using packtools.grid_search GridSearchCVProgressBar make warning ...
# but copied code just work fine, wtf??
# from pactools.grid_search import GridSearchCVProgressBar as _GridSearchCVProgressBar


class RandomizedSearchCVProgressBar(model_selection.RandomizedSearchCV):
    """Monkey patch Parallel to have a progress bar during Randomized search"""

    def _get_param_iterator(self):
        """Return RandomizedSearchCV instance for the given param_grid"""

        iterator = super(RandomizedSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)

        cv = model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits

        class ParallelProgressBar(Parallel):
            def __call__(self, iterable):
                bar = ProgressBar(max_value=max_value, title='GridSearchCV')
                iterable = bar(iterable)
                return super(ParallelProgressBar, self).__call__(iterable)

        # Monkey patch
        model_selection._search.Parallel = ParallelProgressBar

        return iterator


class wrapperRandomizedSearchCV(RandomizedSearchCVProgressBar, ClfWrapperMixIn, metaclass=MetaBaseWrapperClf_with_ABC):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None, fit_params=None, n_jobs=CPU_COUNT,
                 iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score="warn"):
        RandomizedSearchCVProgressBar.__init__(
            self, estimator, param_distributions, n_iter, scoring, fit_params, n_jobs, iid, refit, cv, verbose,
            pre_dispatch, random_state, error_score, return_train_score)
        ClfWrapperMixIn.__init__(self)
