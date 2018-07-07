from sklearn.linear_model import LassoLars as _LassoLars
from sklearn.linear_model import LassoLarsCV as _LassoLarsCV
from sklearn.linear_model import ElasticNet as _ElasticNetReg
from sklearn.linear_model import ElasticNetCV as _ElasticNetCVReg
from sklearn.linear_model import BayesianRidge as _BayesianRidgeReg
from sklearn.linear_model import ARDRegression as _ARDRegression
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.linear_model import HuberRegressor as _HuberRegressor
from sklearn.linear_model import RANSACRegressor as _RANSACRegressor
from sklearn.linear_model import TheilSenRegressor as _TheilSenRegressor
from sklearn.isotonic import IsotonicRegression as _IsotonicRegression
from sklearn.kernel_ridge import KernelRidge as _KernelRidge
from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC
import numpy as np


# TODO
# kernel
# RBF
# martern
# RationalQuadratic
# Dotproduct
# ExpSineSquared


class skLassoLarsReg(_LassoLars, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto',
                 max_iter=500, eps=np.finfo(np.float).eps, copy_X=True, fit_path=True, positive=False):
        _LassoLars.__init__(
            self, alpha, fit_intercept, verbose, normalize, precompute, max_iter, eps, copy_X, fit_path, positive)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'alpha': 1.0,
        'fit_intercept': True,
        'verbose': False,
        'normalize': True,
        'precompute': 'auto',
        'max_iter': 500,
        'eps': np.finfo(np.float).eps,
        'copy_X': True,
        'fit_path': True,
        'positive': False,
    }


class skLassoLarsCVReg(_LassoLarsCV, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto', cv=None,
                 max_n_alphas=1000, n_jobs=1, eps=np.finfo(np.float).eps, copy_X=True, positive=False):
        _LassoLarsCV.__init__(
            self, fit_intercept, verbose, max_iter, normalize, precompute, cv, max_n_alphas, n_jobs, eps, copy_X,
            positive)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'fit_intercept': True,
        'verbose': False,
        'max_iter': 500,
        'normalize': True,
        'precompute': 'auto',
        'cv': None,
        'max_n_alphas': 1000,
        'n_jobs': 1,
        'eps': np.finfo(np.float).eps,
        'copy_X': True,
        'positive': False,
    }


class skElasticNetReg(_ElasticNetReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False,
                 max_iter=1000, copy_X=True, tol=1e-4, warm_start=False, positive=False, random_state=None,
                 selection='cyclic'):
        _ElasticNetReg.__init__(
            self, alpha, l1_ratio, fit_intercept, normalize, precompute, max_iter, copy_X, tol, warm_start,
            positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'ffalpha': 1.0,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'normalize': False,
        'precompute': False,
        'max_iter': 1000,
        'copy_X': True,
        'tol': 1e-4,
        'warm_start': False,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }


class skElasticNetCvReg(_ElasticNetCVReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True, normalize=False,
                 precompute='auto', max_iter=1000, tol=1e-4, cv=None, copy_X=True, verbose=0, n_jobs=1,
                 positive=False, random_state=None, selection='cyclic'):
        _ElasticNetCVReg.__init__(
            self, l1_ratio, eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, cv,
            copy_X, verbose, n_jobs, positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'l1_ratio': 0.5,
        'eps': 1e-3,
        'n_alphas': 100,
        'alphas': None,
        'fit_intercept': True,
        'normalize': False,
        'precompute': 'auto',
        'max_iter': 1000,
        'tol': 1e-4,
        'cv': None,
        'copy_X': True,
        'verbose': 0,
        'n_jobs': 1,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }


class skBayesianRidgeReg(_BayesianRidgeReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6,
                 compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
        _BayesianRidgeReg.__init__(
            self, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score, fit_intercept, normalize,
            copy_X, verbose)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'n_iter': 300,
        'tol': 1.e-3,
        'alpha_1': 1.e-6,
        'alpha_2': 1.e-6,
        'lambda_1': 1.e-6,
        'lambda_2': 1.e-6,
        'compute_score': False,
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'verbose': False,
    }


class skARDReg(_ARDRegression, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6,
                 compute_score=False, threshold_lambda=1.e+4, fit_intercept=True, normalize=False, copy_X=True,
                 verbose=False):
        _ARDRegression.__init__(
            self, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score, threshold_lambda,
            fit_intercept, normalize, copy_X, verbose)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'n_iter': 300,
        'tol': 1.e-3,
        'alpha_1': 1.e-6,
        'alpha_2': 1.e-6,
        'lambda_1': 1.e-6,
        'lambda_2': 1.e-6,
        'compute_score': False,
        'threshold_lambda': 1.e+4,
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'verbose': False,
    }


class skLogisticReg(_LogisticRegression, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                 verbose=0, warm_start=False, n_jobs=1):
        _LogisticRegression.__init__(
            self, penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver,
            max_iter, multi_class, verbose, warm_start, n_jobs)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 1.0,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'class_weight': None,
        'random_state': None,
        'solver': 'liblinear',
        'max_iter': 100,
        'multi_class': 'ovr',
        'verbose': 0,
        'warm_start': False,
        'n_jobs': 1,
    }


class skRANSACReg(_RANSACRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None,
                 is_model_valid=None, max_trials=100, max_skips=np.inf, stop_n_inliers=np.inf, stop_score=np.inf,
                 stop_probability=0.99, residual_metric=None, loss='absolute_loss', random_state=None):
        _RANSACRegressor.__init__(
            self, base_estimator, min_samples, residual_threshold, is_data_valid, is_model_valid, max_trials,
            max_skips, stop_n_inliers, stop_score, stop_probability, residual_metric, loss, random_state)
        BaseWrapperReg.__init__(self)

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


class skTheilSenReg(_TheilSenRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, fit_intercept=True, copy_X=True, max_subpopulation=1e4, n_subsamples=None, max_iter=300,
                 tol=1.e-3, random_state=None, n_jobs=1, verbose=False):
        _TheilSenRegressor.__init__(
            self, fit_intercept, copy_X, max_subpopulation, n_subsamples, max_iter, tol, random_state, n_jobs,
            verbose)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'fit_intercept': True,
        'copy_X': True,
        'max_subpopulation': 1e4,
        'n_subsamples': None,
        'max_iter': 300,
        'tol': 1.e-3,
        'random_state': None,
        'n_jobs': 1,
        'verbose': False,
    }


class skKernelRidgeReg(_KernelRidge, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, alpha=1, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None):
        _KernelRidge.__init__(self, alpha, kernel, gamma, degree, coef0, kernel_params)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'alpha': 1,
        'kernel': "linear",
        'gamma': None,
        'degree': 3,
        'coef0': 1,
        'kernel_params': None,
    }


class skHuberReg(_HuberRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
        _HuberRegressor.__init__(self, epsilon, max_iter, alpha, warm_start, fit_intercept, tol)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'epsilon': 1.35,
        'max_iter': 100,
        'alpha': 0.0001,
        'warm_start': False,
        'fit_intercept': True,
        'tol': 1e-05,
    }


class skIsotonicReg(_IsotonicRegression, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, y_min=None, y_max=None, increasing=True, out_of_bounds='nan'):
        _IsotonicRegression.__init__(self, y_min, y_max, increasing, out_of_bounds)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'y_min': None,
        'y_max': None,
        'increasing': True,
        'out_of_bounds': 'nan',
    }
