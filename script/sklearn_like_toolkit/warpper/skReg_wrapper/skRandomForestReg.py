from hyperopt import hp
from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skRandomForestReg(_RandomForestRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 warm_start=False):
        n_jobs = 4
        n_estimators = int(n_estimators)

        _RandomForestRegressor.__init__(
            self, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'n_estimators': hp.qloguniform('n_estimators', 2, 5, 1),
        'max_depth': 1 + hp.randint('max_depth', 10),
        'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
        'min_samples_split': 2 + hp.randint('min_samples_split', 10),
        'criterion': hp.choice('criterion', ["mae", 'mse']),
    }

    tuning_grid = {
        'n_estimators': 10,
        'criterion': "mse",
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.,
        'max_features': "auto",
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': 1,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }