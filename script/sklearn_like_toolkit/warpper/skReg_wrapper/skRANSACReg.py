import numpy as np
from sklearn.linear_model import RANSACRegressor as _RANSACRegressor

from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC


class skRANSACReg(_RANSACRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None,
                 is_model_valid=None, max_trials=100, max_skips=np.inf, stop_n_inliers=np.inf, stop_score=np.inf,
                 stop_probability=0.99, residual_metric=None, loss='absolute_loss', random_state=None):
        _RANSACRegressor.__init__(
            self, base_estimator, min_samples, residual_threshold, is_data_valid, is_model_valid, max_trials,
            max_skips, stop_n_inliers, stop_score, stop_probability, residual_metric, loss, random_state)
        BaseWrapperReg.__init__(self)

    # TODO
    HyperOpt_space = {}

    tuning_grid = {
        'base_estimator': None,
        'min_samples': None,
        'residual_threshold': None,
        'is_data_valid': None,
        'is_model_valid': None,
        'max_trials': 100,
        'max_skips': np.inf,
        'stop_n_inliers': np.inf,
        'stop_score': np.inf,
        'stop_probability': 0.99,
        'residual_metric': None,
        'loss': 'absolute_loss',
        'random_state': None,
    }