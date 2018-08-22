from hyperopt import hp
from sklearn.tree import DecisionTreeRegressor as _DecisionTreeRegressor

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skDecisionTreeReg(_DecisionTreeRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0., min_impurity_split=None, presort=False):
        _DecisionTreeRegressor.__init__(
            self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, presort)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'max_depth': 1 + hp.randint('max_depth', 10),
        'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
        'min_samples_split': 2 + hp.randint('min_samples_split', 10),
        'criterion': hp.choice('criterion', ['mse', 'mae']),
        'splitter': hp.choice('splitter', ['best', 'random']),
    }

    tuning_grid = {
        'criterion': "mse",
        'splitter': "best",
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.,
        'max_features': None,
        'random_state': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.,
        'min_impurity_split': None,
        'presort': False,
    }
