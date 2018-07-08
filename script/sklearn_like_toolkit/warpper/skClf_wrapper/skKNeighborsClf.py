from hyperopt import hp
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC


class skKNeighborsClf(_KNeighborsClassifier, BaseWrapperClf, metaclass=meta_BaseWrapperClf_with_ABC):

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=1, **kwargs):
        leaf_size = int(leaf_size)
        n_neighbors = int(n_neighbors)
        _KNeighborsClassifier.__init__(
            self, n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs, **kwargs)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'n_neighbors': 1 + hp.randint('n_neighbors', 50),
        'leaf_size': hp.qloguniform('leaf_size', 2, 5, 1),
        'p': hp.uniform('p', 1, 2),

        # 'metric': 'minkowski',
        # 'metric_params': None,

        # 'n_jobs': 1,
        # 'weights': 'uniform',
        # 'algorithm': 'auto',
    }
    tuning_grid = {
        'n_neighbors': 5,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,
        'metric': 'minkowski',
        'metric_params': None,
        'n_jobs': 1,
    }
