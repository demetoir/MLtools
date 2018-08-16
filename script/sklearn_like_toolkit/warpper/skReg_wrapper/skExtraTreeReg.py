from hyperopt import hp
from sklearn.tree import ExtraTreeRegressor as _ExtraTreeRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperReg_with_ABC


class skExtraTreeReg(_ExtraTreeRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):

    def __init__(self, criterion="mse", splitter="random", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", random_state=None, min_impurity_decrease=0.,
                 min_impurity_split=None, max_leaf_nodes=None):
        _ExtraTreeRegressor.__init__(
            self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, min_impurity_decrease, min_impurity_split, max_leaf_nodes)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'max_depth': 1 + hp.randint('max_depth', 10),
        'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
        'min_samples_split': 2 + hp.randint('min_samples_split', 10),
        'criterion': hp.choice('criterion', ['mse', 'mae']),
        'splitter': hp.choice('splitter', ['random', 'best']),
    }

    tuning_grid = {
        'criterion': "mse",
        'splitter': "random",
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.,
        'max_features': "auto",
        'random_state': None,
        'min_impurity_decrease': 0.,
        'min_impurity_split': None,
        'max_leaf_nodes': None,
    }