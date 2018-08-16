from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import MetaBaseWrapperClf_with_ABC, MetaBaseWrapperReg_with_ABC
import warnings
import lightgbm
from hyperopt import hp


class LightGBMClf(lightgbm.LGBMClassifier, BaseWrapperClf, metaclass=MetaBaseWrapperClf_with_ABC):
    HyperOpt_space = {
        'boosting_type': hp.choice('boosting_type', ['dart', "gbdt"]),
        'max_depth': 2 + hp.randint('max_depth', 10),
        'n_estimators': 10 + hp.randint('n_estimators', 400),
        'subsample': hp.uniform('subsample', 0, 1),
        'min_child_samples': hp.qloguniform('min_child_samples', 2, 4, 1),
        'num_leaves': hp.qloguniform('num_leaves', 2, 5, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
    }
    tuning_grid = {
        'num_leaves': [4, 8, 16, 32],
        'min_child_samples': [4, 8, 16, 32],
        'max_depth': [2, 4, 6, 8],
        'colsample_bytree': [i / 10.0 for i in range(2, 10 + 1, 2)],
        'subsample': [i / 10.0 for i in range(2, 10 + 1, 2)],
        'max_bin': [64, 128],
        # 'top_k': [8, 16, 32],
        # 'learning_rate': 0.1,
    }
    remain_param = {
        'learning_rate': 0.1,
        # 'num_boost_round': 100,
        # ???
        'max_delta_step': 0,
        'min_split_gain': 0,

        # device option
        # 'device': 'gpu',
        'device': 'cpu',

        # dart only option
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'max_drop': 50,
        'uniform_drop': False,
        'xgboost_dart_mode': False,
        'drop_seed': 4,

        # goss only option
        'other_rate': 0.1,
        'min_data_per_group': 100,

        # default value

        'bagging_seed': 3,

        # 'early_stopping_round': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'max_cat_threshold': 32,
        'cat_smooth': 10,
        'cat_l2': 10,
        'max_cat_to_onehot': 4,
        'verbose': -1

    }

    def __init__(
            self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100,
            subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0., min_child_weight=1e-3,
            min_child_samples=20, subsample=1., subsample_freq=1, colsample_bytree=1., reg_alpha=0., reg_lambda=0.,
            random_state=None, n_jobs=-1, silent=True, **kwargs):
        kwargs['verbose'] = -1

        num_leaves = int(num_leaves)
        min_child_samples = int(min_child_samples)
        n_estimators = int(n_estimators)
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        lightgbm.LGBMClassifier.__init__(
            self, boosting_type, num_leaves, max_depth, learning_rate, n_estimators, subsample_for_bin, objective,
            class_weight, min_split_gain, min_child_weight, min_child_samples, subsample, subsample_freq,
            colsample_bytree, reg_alpha, reg_lambda, random_state, n_jobs, silent, **kwargs)

        BaseWrapperClf.__init__(self)

    @property
    def feature_importances(self):
        return self.feature_importances_


class LightGBMReg(lightgbm.LGBMRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperReg_with_ABC):
    HyperOpt_space = {
        'boosting_type': hp.choice('boosting_type', ['dart', "gbdt"]),
        'max_depth': 2 + hp.randint('max_depth', 10),
        'n_estimators': 10 + hp.randint('n_estimators', 400),
        'subsample': hp.uniform('subsample', 0, 1),
        'min_child_samples': hp.qloguniform('min_child_samples', 2, 4, 1),
        'num_leaves': hp.qloguniform('num_leaves', 2, 5, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
    }
    tuning_grid = {
        # 'num_leaves': [4, 8, 16, 32],
        # 'min_child_samples': [4, 8, 16, 32],
        # 'max_depth': [2, 4, 6, 8],
        # 'colsample_bytree': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'subsample': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'max_bin': [64, 128],1
        # 'top_k': [8, 16, 32],
    }
    remain_param = {
        'learning_rate': 0.1,
        # 'num_boost_round': 100,
        # ???
        'max_delta_step': 0,
        'min_split_gain': 0,

        # device option
        # 'device': 'gpu',
        'device': 'cpu',

        # dart only option
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'max_drop': 50,
        'uniform_drop': False,
        'xgboost_dart_mode': False,
        'drop_seed': 4,

        # goss only option
        'other_rate': 0.1,
        'min_data_per_group': 100,

        # default value
        'feature_fraction_seed': 2,
        'bagging_seed': 3,
        # 'early_stopping_round': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'max_cat_threshold': 32,
        'cat_smooth': 10,
        'cat_l2': 10,
        'max_cat_to_onehot': 4,
        'verbose': -1

    }

    def __init__(
            self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100,
            subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0., min_child_weight=1e-3,
            min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0.,
            random_state=None, n_jobs=-1, silent=True, **kwargs):
        kwargs['verbose'] = -1
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        warnings.filterwarnings(module='lightgbm*', action='ignore', category=UserWarning)
        num_leaves = int(num_leaves)
        min_child_samples = int(min_child_samples)

        lightgbm.LGBMRegressor.__init__(
            self, boosting_type, num_leaves, max_depth, learning_rate, n_estimators, subsample_for_bin, objective,
            class_weight, min_split_gain, min_child_weight, min_child_samples, subsample, subsample_freq,
            colsample_bytree, reg_alpha, reg_lambda, random_state, n_jobs, silent, **kwargs)

        BaseWrapperReg.__init__(self)


    @property
    def feature_importances(self):
        return self.feature_importances_
