from hyperopt import hp
from sklearn.neighbors import NearestCentroid as _NearestCentroid

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperClf_with_ABC


class skNearestCentroidClf(_NearestCentroid, BaseWrapperClf, metaclass=MetaBaseWrapperClf_with_ABC):

    def __init__(self, metric='euclidean', shrink_threshold=None):
        _NearestCentroid.__init__(self, metric, shrink_threshold)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'metric': hp.choice('metric', ['euclidean', 'manhattan']),
        'shrink_threshold': hp.uniform('shrink_threshold', 0, 1),
    }

    tuning_grid = {
        'metric': ['euclidean', 'manhattan'],
        'shrink_threshold': None,
    }

