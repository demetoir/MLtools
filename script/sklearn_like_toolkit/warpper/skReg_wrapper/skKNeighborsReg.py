from hyperopt import hp
from sklearn.neighbors import KNeighborsRegressor as _KNeighborsRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skKNeighborsReg(_KNeighborsRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=1, **kwargs):
        n_jobs = 4
        _KNeighborsRegressor.__init__(
            self, n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs, **kwargs)
        BaseWrapperReg.__init__(self)

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