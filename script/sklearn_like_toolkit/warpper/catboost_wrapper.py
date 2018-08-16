import warnings
from catboost import CatBoostClassifier as _CatBoostClassifier
from catboost import CatBoostRegressor as _CatBoostRegressor
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperClf_with_ABC, MetaBaseWrapperReg_with_ABC
from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from hyperopt import hp


class CatBoostClf(BaseWrapperClf, _CatBoostClassifier, metaclass=MetaBaseWrapperClf_with_ABC):
    HyperOpt_space = {
        'iterations': 2 + hp.randint('iterations', 10),
        'depth': 4 + hp.randint('depth', 11),
        'random_strength': hp.choice('random_strength', [1, 2, 4, 0.5, ]),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
        'learning_rate': hp.loguniform('learning_rate', -6, 0),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 1),
    }
    tuning_grid = {
        'iterations': [2, 4, 8, ],
        'depth': [i for i in range(4, 10 + 1, 2)],
        # 'random_strength': [1, 2, 4, 0.5, ],
        'bagging_temperature': [i / 100.0 for i in range(1, 10 + 1, 3)],
        'learning_rate': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'l2_leaf_reg': [i / 10.0 for i in range(1, 10 + 1, 3)],

    }
    remain_param = {
        'use_best_model': [True, False],
        'eval_metric': [],
        'od_type': None,
        'od_pval': None,
        'od_wait': None
    }

    def __init__(self, **kwargs):
        # logging_level = 'Silent'
        # silent = True
        # kwargs['silent'] = True
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        BaseWrapperClf.__init__(self)
        _CatBoostClassifier.__init__(self, **kwargs)

    @property
    def feature_importances(self):
        return self.feature_importances_


class CatBoostReg(_CatBoostRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):
    HyperOpt_space = {
        'iterations': 2 + hp.randint('iterations', 10),
        'depth': 4 + hp.randint('depth', 11),
        'random_strength': hp.choice('random_strength', [1, 2, 4, 0.5, ]),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
        'learning_rate': hp.loguniform('learning_rate', -6, 0),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 1),
    }
    tuning_grid = {
        # 'iterations': [2, 4, 8, ],
        # 'depth': [i for i in range(4, 10 + 1, 2)],
        # # 'random_strength': [1, 2, 4, 0.5, ],
        # 'bagging_temperature': [i / 100.0 for i in range(1, 10 + 1, 3)],
        # 'learning_rate': [i / 10.0 for i in range(1, 10 + 1, 3)],
        # 'l2_leaf_reg': [i / 10.0 for i in range(1, 10 + 1, 3)],

    }
    remain_param = {
        'use_best_model': [True, False],
        'eval_metric': [],
        'od_type': None,
        'od_pval': None,
        'od_wait': None
    }

    def __init__(self, **kwargs):
        # silent = True
        _CatBoostRegressor.__init__(self, **kwargs)
        BaseWrapperReg.__init__(self)

    @property
    def feature_importances(self):
        return self.feature_importances_
