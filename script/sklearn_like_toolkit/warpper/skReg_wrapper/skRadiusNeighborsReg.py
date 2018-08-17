from hyperopt import hp
from sklearn.neighbors import RadiusNeighborsRegressor as _RadiusNeighborsRegressor

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skRadiusNeighborsReg(_RadiusNeighborsRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, **kwargs):
        _RadiusNeighborsRegressor.__init__(
            self, radius, weights, algorithm, leaf_size, p, metric, metric_params, **kwargs)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'radius': hp.loguniform('radius', -1, 3),
        'leaf_size': hp.qloguniform('leaf_size', 2, 5, 1),
        'p': hp.uniform('p', 1, 2),
    }
    tuning_grid = {
        'radius': 1.0,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,
        'metric': 'minkowski',
        'outlier_label': None,
        'metric_params': None,
    }
