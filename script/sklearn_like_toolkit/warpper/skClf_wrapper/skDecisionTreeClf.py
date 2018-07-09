from hyperopt import hp
from sklearn.tree import DecisionTreeClassifier as _skDecisionTreeClassifier
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC


class skDecisionTreeClf(BaseWrapperClf, _skDecisionTreeClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0., min_impurity_split=None, class_weight=None, presort=False):
        _skDecisionTreeClassifier.__init__(
            self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight,
            presort)
        BaseWrapperClf.__init__(self)

    HyperOpt_space = {
        'max_depth': 1 + hp.randint('max_depth', 10),
        'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
        'min_samples_split': 2 + hp.randint('min_samples_split', 10),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'splitter': hp.choice('splitter', ['best', 'random']),

        # 'max_leaf_nodes': None,
        # 'max_features': None,
        # 'min_impurity_decrease': 0.0,
        # 'class_weight': None,
        # 'min_weight_fraction_leaf': 0.0,
    }

    tuning_grid = {
        'max_depth': [i for i in range(1, 10)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    etc_param = {
        # class weight options

        'presort': False,
        'random_state': None,
    }
