from sklearn.linear_model import Lasso as _LassoReg
from sklearn.linear_model import LassoCV as _LassoCVReg
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
from sklearn.linear_model import SGDRegressor as _SGDRegressor
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier as _PassiveAggressiveClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor as _PassiveAggressiveRegressor
from sklearn.linear_model.ridge import Ridge as _RidgeReg
from sklearn.linear_model.ridge import RidgeCV as _RidgeCVReg
from sklearn.linear_model.stochastic_gradient import DEFAULT_EPSILON
from sklearn.neighbors import RadiusNeighborsClassifier as _RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor as _RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor as _KNeighborsRegressor
from sklearn.neighbors import NearestCentroid as _NearestCentroid
from sklearn.neural_network import MLPRegressor as _MLPRegressor
from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as _GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor as _BaggingRegressor
from sklearn.tree import ExtraTreeRegressor as _ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor as _DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression as _IsotonicRegression
from sklearn.kernel_ridge import KernelRidge as _KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor as _GaussianProcessRegressor
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC, \
    meta_BaseWrapperReg_with_ABC
import numpy as np


# TODO
# kernel
# RBF
# martern
# RationalQuadratic
# Dotproduct
# ExpSineSquared


class skPassiveAggressiveClf(_PassiveAggressiveClassifier, BaseWrapperClf, metaclass=meta_BaseWrapperClf_with_ABC):

    def __init__(self, C=1.0, fit_intercept=True, max_iter=1000, tol=None, shuffle=True, verbose=0, loss="hinge",
                 n_jobs=1, random_state=None, warm_start=False, class_weight=None, average=False, n_iter=None):
        _PassiveAggressiveClassifier.__init__(
            self, C, fit_intercept, max_iter, tol, shuffle, verbose, loss, n_jobs, random_state, warm_start,
            class_weight, average, n_iter)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'C': 1.0,
        'fit_intercept': True,
        'max_iter': None,
        'tol': None,
        'shuffle': True,
        'verbose': 0,
        'loss': "hinge",
        'n_jobs': 1,
        'random_state': None,
        'warm_start': False,
        'class_weight': None,
        'average': False,
        'n_iter': None,
    }


class skRadiusNeighborsClf(_RadiusNeighborsClassifier, BaseWrapperClf, metaclass=meta_BaseWrapperClf_with_ABC):

    def __init__(self, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 outlier_label=None, metric_params=None, **kwargs):
        _RadiusNeighborsClassifier.__init__(
            self, radius, weights, algorithm, leaf_size, p, metric, outlier_label, metric_params, **kwargs)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'radius': 1.0,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,
        'metric': 'minkowski',
        'outlier_label': None,
        'metric_params': None,
    }


class skNearestCentroidClf(_NearestCentroid, BaseWrapperClf, metaclass=meta_BaseWrapperClf_with_ABC):

    def __init__(self, metric='euclidean', shrink_threshold=None):
        _NearestCentroid.__init__(self, metric, shrink_threshold)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'metric': 'euclidean',
        'shrink_threshold': None,
    }


class skRidgeReg(_RidgeReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-3,
                 solver="auto", random_state=None):
        _RidgeReg.__init__(self, alpha, fit_intercept, normalize, copy_X, max_iter, tol, solver, random_state)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'alpha': 1.0,
        'tol': 1e-3,

        # 'fit_intercept': True,
        # 'normalize': False,
        # 'max_iter': None,

        # 'copy_X': True,
        # 'solver': "auto",
        # 'random_state': None,
    }


class skRidgeCVReg(_RidgeCVReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=None,
                 gcv_mode=None, store_cv_values=False):
        _RidgeCVReg.__init__(self, alphas, fit_intercept, normalize, scoring, cv, gcv_mode, store_cv_values)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        # shape positive float
        'alphas': (0.1, 1.0, 10.0),
        # 'fit_intercept': True,
        # 'normalize': False,
        # 'scoring': None,
        # 'cv': None,
        # 'class_weight': None
    }


class skLassoReg(_LassoReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        _LassoReg.__init__(
            self, alpha, fit_intercept, normalize, precompute, copy_X, max_iter, tol, warm_start, positive,
            random_state, selection)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'alpha': 1.0,
        'fit_intercept': True,
        'normalize': False,
        'precompute': False,
        'copy_X': True,
        'max_iter': 1000,
        'tol': 1e-4,
        'warm_start': False,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }


class skLassoCVReg(_LassoCVReg, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, copy_X=True, cv=None, verbose=False, n_jobs=1, positive=False,
                 random_state=None, selection='cyclic'):
        _LassoCVReg.__init__(
            self, eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, copy_X, cv,
            verbose, n_jobs, positive, random_state, selection)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'eps': 1e-3,
        'n_alphas': 100,
        'alphas': None,
        'fit_intercept': True,
        'normalize': False,
        'precompute': 'auto',
        'max_iter': 1000,
        'tol': 1e-4,
        'copy_X': True,
        'cv': None,
        'verbose': False,
        'n_jobs': 1,
        'positive': False,
        'random_state': None,
        'selection': 'cyclic',
    }


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


class skSGDReg(_SGDRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                 max_iter=1000, tol=None, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.25, warm_start=False, average=False, n_iter=None):
        _SGDRegressor.__init__(
            self, loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon,
            random_state, learning_rate, eta0, power_t, warm_start, average, n_iter)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'loss': "squared_loss",
        'penalty': "l2",
        'alpha': 0.0001,
        'l1_ratio': 0.15,
        'fit_intercept': True,
        'max_iter': None,
        'tol': None,
        'shuffle': True,
        'verbose': 0,
        'epsilon': DEFAULT_EPSILON,
        'random_state': None,
        'learning_rate': "invscaling",
        'eta0': 0.01,
        'power_t': 0.25,
        'warm_start': False,
        'average': False,
        'n_iter': None,
    }


class skPassiveAggressiveReg(_PassiveAggressiveRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, C=1.0, fit_intercept=True, max_iter=1000, tol=None, shuffle=True, verbose=0,
                 loss="epsilon_insensitive", epsilon=DEFAULT_EPSILON, random_state=None, warm_start=False,
                 average=False, n_iter=None):
        _PassiveAggressiveRegressor.__init__(
            self, C, fit_intercept, max_iter, tol, shuffle, verbose, loss, epsilon, random_state, warm_start,
            average, n_iter)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'C': 1.0,
        'fit_intercept': True,
        'max_iter': None,
        'tol': None,
        'shuffle': True,
        'verbose': 0,
        'loss': "epsilon_insensitive",
        'epsilon': DEFAULT_EPSILON,
        'random_state': None,
        'warm_start': False,
        'average': False,
        'n_iter': None,
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


class skRadiusNeighborsReg(_RadiusNeighborsRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, **kwargs):
        _RadiusNeighborsRegressor.__init__(
            self, radius, weights, algorithm, leaf_size, p, metric, metric_params, **kwargs)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'radius': 1.0,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,
        'metric': 'minkowski',
        'outlier_label': None,
        'metric_params': None,
    }


class skKNeighborsReg(_KNeighborsRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=1, **kwargs):
        _KNeighborsRegressor.__init__(
            self, n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs, **kwargs)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'n_neighbors': 5,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,
        'metric': 'minkowski',
        'metric_params': None,
        'n_jobs': 1,
    }


class skGaussianProcessReg(_GaussianProcessRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        _GaussianProcessRegressor.__init__(
            self, kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train, random_state)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'kernel': None,
        'alpha': 1e-10,
        'optimizer': "fmin_l_bfgs_b",
        'n_restarts_optimizer': 0,
        'normalize_y': False,
        'copy_X_train': True,
        'random_state': None,
    }


class skDecisionTreeReg(_DecisionTreeRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0., min_impurity_split=None, presort=False):
        _DecisionTreeRegressor.__init__(
            self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, presort)
        BaseWrapperReg.__init__(self)

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


class skExtraTreeReg(_ExtraTreeRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, criterion="mse", splitter="random", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", random_state=None, min_impurity_decrease=0.,
                 min_impurity_split=None, max_leaf_nodes=None):
        _ExtraTreeRegressor.__init__(
            self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, min_impurity_decrease, min_impurity_split, max_leaf_nodes)
        BaseWrapperReg.__init__(self)

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


class skAdaBoostReg(_AdaBoostRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1., loss='linear', random_state=None):
        _AdaBoostRegressor.__init__(self, base_estimator, n_estimators, learning_rate, loss, random_state)

    tuning_grid = {
        'base_estimator': None,
        'n_estimators': 50,
        'learning_rate': 1.,
        'loss': 'linear',
        'random_state': None,
    }


class skBaggingReg(_BaggingRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None,
                 verbose=0):
        _BaggingRegressor.__init__(
            self, base_estimator, n_estimators, max_samples, max_features, bootstrap, bootstrap_features,
            oob_score, warm_start, n_jobs, random_state, verbose)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'base_estimator': None,
        'n_estimators': 10,
        'max_samples': 1.0,
        'max_features': 1.0,
        'bootstrap': True,
        'bootstrap_features': False,
        'oob_score': False,
        'warm_start': False,
        'n_jobs': 1,
        'random_state': None,
        'verbose': 0,
    }


class skRandomForestReg(_RandomForestRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 warm_start=False):
        _RandomForestRegressor.__init__(
            self, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start)
        BaseWrapperReg.__init__(self)

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


class skGradientBoostingReg(_GradientBoostingRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3,
                 min_impurity_decrease=0., min_impurity_split=None, init=None, random_state=None, max_features=None,
                 alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'):
        _GradientBoostingRegressor.__init__(
            self, loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_depth, min_impurity_decrease, min_impurity_split, init, random_state,
            max_features, alpha, verbose, max_leaf_nodes, warm_start, presort)
        BaseWrapperReg.__init__(self)

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


class skMLPReg(_MLPRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        _MLPRegressor.__init__(
            self, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init,
            power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum,
            nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon)
        BaseWrapperReg.__init__(self)

    tuning_grid = {
        'hidden_layer_sizes': (100,),
        'activation': "relu",
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': "constant",
        'learning_rate_init': 0.001,
        'power_t': 0.5,
        'max_iter': 200,
        'shuffle': True,
        'random_state': None,
        'tol': 1e-4,
        'verbose': False,
        'warm_start': False,
        'momentum': 0.9,
        'nesterovs_momentum': True,
        'early_stopping': False,
        'validation_fraction': 0.1,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8,
    }
