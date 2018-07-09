from hyperopt import hp
from sklearn.neighbors import RadiusNeighborsClassifier as _RadiusNeighborsClassifier

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC


class skRadiusNeighborsClf(_RadiusNeighborsClassifier, BaseWrapperClf, metaclass=meta_BaseWrapperClf_with_ABC):

    def __init__(self, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 outlier_label=None, metric_params=None, **kwargs):
        _RadiusNeighborsClassifier.__init__(
            self, radius, weights, algorithm, leaf_size, p, metric, outlier_label, metric_params, **kwargs)
        BaseWrapperClf.__init__(self)

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
