from hyperopt import hp
from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skAdaBoostReg(_AdaBoostRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1., loss='linear', random_state=None):
        n_estimators = int(n_estimators)
        _AdaBoostRegressor.__init__(self, base_estimator, n_estimators, learning_rate, loss, random_state)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'n_estimators': hp.qloguniform('n_estimators', 2, 6, 1),
        'learning_rate': hp.loguniform('learning_rate', -8, 1),
        'loss': hp.choice('loss', ['linear', 'square', 'exponential']),
    }

    tuning_grid = {
        'base_estimator': None,
        'n_estimators': 50,
        'learning_rate': 1.,
        'loss': 'linear',
        'random_state': None,
    }