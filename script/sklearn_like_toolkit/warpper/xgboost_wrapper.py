from hyperopt import hp

from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC, meta_BaseWrapperReg_with_ABC
from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
import warnings
import xgboost as xgb
import numpy as np


class XGBoostClf(xgb.XGBClassifier, BaseWrapperClf, metaclass=meta_BaseWrapperClf_with_ABC):
    HyperOpt_space = {
        'n_estimators': 10 + hp.randint('n_estimators', 400),
        'max_depth': 4 + hp.randint('max_depth', 11),
        'min_child_weight': 1 + hp.randint('min_child_weight', 3),
        'gamma': hp.uniform('gamma', 0, 1),
        'subsample': hp.uniform('subsample', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
        'learning_rate': hp.loguniform('learning_rate', -6, 0),
    }

    tuning_grid = {
        'max_depth': [4, 6, 8],
        # 'n_estimators': [128, 256],
        # 'min_child_weight': [1, 2, 3],
        'gamma': [i / 10.0 for i in range(2, 10 + 1, 2)],
        'subsample': [i / 10.0 for i in range(2, 10 + 1, 2)],
        'colsample_bytree': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'learning_rate': [0.01, 0.1, 1],
    }
    tuning_params = {
        'max_depth': 3,
        'n_estimators': 100,
        'min_child_weight': 1,
        'gamma': 0,

        'subsample': 1,
        'colsample_bytree': 1,
        'learning_rate': 0.1,
    }
    remain_param = {
        'silent': True,
        'objective': 'binary:logistic',
        'booster': ['gbtree', 'gblinear', 'dart'],
        'colsample_bylevel': 1,

        'reg_alpha': 0,
        'reg_lambda': 1,

        'scale_pos_weight': 1,
        'max_delta_step': 0,

        'base_score': 0.5,
        'n_jobs': 1,
        'nthread': None,
        'random_state': 0,
        'seed': None,
        'missing': None,
    }

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="binary:logistic",
                 booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                 random_state=0, seed=None, missing=None, **kwargs):
        xgb.XGBClassifier.__init__(self, max_depth, learning_rate, n_estimators, silent, objective, booster, n_jobs,
                                   nthread, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree,
                                   colsample_bylevel, reg_alpha, reg_lambda, scale_pos_weight, base_score, random_state,
                                   seed, missing, **kwargs)
        BaseWrapperClf.__init__(self)
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        # params.update({"tree_method": 'auto'})
        # params.update({"tree_method": 'gpu_hist'})
        # params.update({"tree_method": 'hist'})
        # params.update({"tree_method": 'exact'})
        # params.update({"tree_method": 'gpu_exact'})
        # params.update({'nthread': 1})
        # params.update({"silent": 1})

    @property
    def feature_importances(self):
        return self.feature_importances_


class XGBoostReg(xgb.XGBRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    HyperOpt_space = {
        'n_estimators': 10 + hp.randint('n_estimators', 400),
        'max_depth': 4 + hp.randint('max_depth', 11),
        'min_child_weight': 1 + hp.randint('min_child_weight', 3),
        'gamma': hp.uniform('gamma', 0, 1),
        'subsample': hp.uniform('subsample', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
        'learning_rate': hp.loguniform('learning_rate', -6, 0),
    }

    tuning_grid = {
        # 'max_depth': [4, 6, 8],
        # 'n_estimators': [128, 256],
        # 'min_child_weight': [1, 2, 3],
        # 'gamma': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'subsample': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'colsample_bytree': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'learning_rate': [0.01, 0.1, 1],
    }
    remain_param = {
        'silent': True,
        'objective': 'binary:logistic',
        'booster': ['gbtree', 'gblinear', 'dart'],
        'colsample_bylevel': 1,

        'reg_alpha': 0,
        'reg_lambda': 1,

        'scale_pos_weight': 1,
        'max_delta_step': 0,

        'base_score': 0.5,
        'n_jobs': 1,
        'nthread': None,
        'random_state': 0,
        'seed': None,
        'missing': None,
    }

    def __init__(
            self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="reg:linear",
            booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
            colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
            random_state=0, seed=None, missing=None, **kwargs):
        # warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

        xgb.XGBRegressor.__init__(
            self, max_depth, learning_rate, n_estimators, silent, objective, booster, n_jobs, nthread, gamma,
            min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel, reg_alpha,
            reg_lambda, scale_pos_weight, base_score, random_state, seed, missing, **kwargs)

        BaseWrapperReg.__init__(self)
        # params.update({"tree_method": 'auto'})
        # params.update({"tree_method": 'gpu_hist'})
        # params.update({"tree_method": 'hist'})
        # params.update({"tree_method": 'exact'})
        # params.update({"tree_method": 'gpu_exact'})
        # params.update({'nthread': 1})
        # params.update({"silent": 1})

    @property
    def feature_importances(self):
        return self.feature_importances_
