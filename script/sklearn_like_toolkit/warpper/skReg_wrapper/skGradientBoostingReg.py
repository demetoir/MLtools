from hyperopt import hp
from sklearn.ensemble import GradientBoostingRegressor as _GradientBoostingRegressor

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skGradientBoostingReg(_GradientBoostingRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3,
                 min_impurity_decrease=0., min_impurity_split=None, init=None, random_state=None, max_features=None,
                 alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'):
        n_estimators = int(n_estimators)
        _GradientBoostingRegressor.__init__(
            self, loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_depth, min_impurity_decrease, min_impurity_split, init, random_state,
            max_features, alpha, verbose, max_leaf_nodes, warm_start, presort)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'learning_rate': hp.loguniform('learning_rate', -8, 1),
        'n_estimators': hp.qloguniform('n_estimators', 2, 5, 1),
        'max_depth': 1 + hp.randint('max_depth', 10),
        'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
        'min_samples_split': 2 + hp.randint('min_samples_split', 10),
        'loss': hp.choice('loss', ['ls', 'lad', 'huber', 'quantile']),
    }
    tuning_grid = {
        'loss': 'ls',
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 1.0,
        'criterion': 'friedman_mse',
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.,
        'max_depth': 3,
        'min_impurity_decrease': 0.,
        'min_impurity_split': None,
        'init': None,
        'random_state': None,
        'max_features': None,
        'alpha': 0.9,
        'verbose': 0,
        'max_leaf_nodes': None,
        'warm_start': False,
        'presort': 'auto',
    }

    def _make_estimator(self, append=True):
        pass