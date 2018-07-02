import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF as _RBF
from sklearn.linear_model.stochastic_gradient import DEFAULT_EPSILON
from sklearn.neural_network import MLPClassifier as _skMLPClassifier
from sklearn.naive_bayes import GaussianNB as _skGaussianNB
from sklearn.naive_bayes import BernoulliNB as _skBernoulliNB
from sklearn.naive_bayes import MultinomialNB as _skMultinomialNB
from sklearn.linear_model import SGDClassifier as _skSGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier as _skGaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier as _skKNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier as _skAdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier as _skExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier as _skRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as _skGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier as _skVotingClassifier
from sklearn.ensemble import BaggingClassifier as _BaggingClassifier
from sklearn.tree import DecisionTreeClassifier as _skDecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _skQDA
from sklearn.svm import LinearSVC as _skLinearSVC
from sklearn.svm import SVC as _skSVC
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf, meta_BaseWrapperClf_with_ABC


class skMLP(BaseWrapperClf, _skMLPClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        warnings.filterwarnings(module='sklearn*', action='ignore', category=ConvergenceWarning)

        BaseWrapperClf.__init__(self)
        _skMLPClassifier.__init__(
            self, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t,
            max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping,
            validation_fraction, beta_1, beta_2, epsilon)

    tuning_grid = {
        'activation': ['relu'],
        'alpha': [0.01, 0.1, 1, 10],
        'hidden_layer_sizes': [(32,), (64,), (128,)],

        # 'max_iter': 200,
        # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
        # 'learning_rate_init': 0.001,
        # 'tol': 0.0001,
    }

    solver_param = {
        'solver': ['lbfgs', 'sgd', 'adam'],

        # adam solver option
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08,

        # sgd solver option
        # between 0 and 1.
        'momentum': 0.9,
        'nesterovs_momentum': True,
        'power_t': 0.5,
    }

    etc_param = {
        'random_state': None,
        'verbose': False,
        'warm_start': False,
        'early_stopping': False,

        # batch option
        'batch_size': 'auto',
        'shuffle': True,
        'validation_fraction': 0.1,
    }


class skGaussian_NB(BaseWrapperClf, _skGaussianNB, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, priors=None):
        _skGaussianNB.__init__(self, priors)
        BaseWrapperClf.__init__(self)

    tuning_grid = {}
    tuning_params = {
        'priors': None
    }


class skBernoulli_NB(BaseWrapperClf, _skBernoulliNB, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True, class_prior=None):
        _skBernoulliNB.__init__(self, alpha, binarize, fit_prior, class_prior)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'binarize': [i / 10.0 for i in range(0, 10)],
        # 'class_prior': None,
        # 'fit_prior': True
    }


class skMultinomial_NB(BaseWrapperClf, _skMultinomialNB, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        _skMultinomialNB.__init__(self, alpha, fit_prior, class_prior)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        # 'class_prior': None,
        # 'fit_prior': True
    }


class skQDA(BaseWrapperClf, _skQDA, metaclass=meta_BaseWrapperClf):
    def __init__(self, priors=None, reg_param=0., store_covariance=False, tol=1.0e-4, store_covariances=None):
        warnings.filterwarnings(module='sklearn*', action='ignore', category=Warning)

        BaseWrapperClf.__init__(self)
        _skQDA.__init__(self, priors, reg_param, store_covariance, tol, store_covariances)

    tuning_grid = {
    }
    remain_param = {
        # TODO
        # ? ..
        'priors': None,
        'reg_param': 0.0,
        'store_covariance': False,
        'store_covariances': None,
        'tol': 0.0001
    }


class skDecisionTree(BaseWrapperClf, _skDecisionTreeClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0., min_impurity_split=None, class_weight=None, presort=False):
        _skDecisionTreeClassifier.__init__(
            self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight,
            presort)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'max_depth': [i for i in range(1, 10)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    default_only_params = {
        'criterion': 'gini',
        'splitter': 'best',
        'max_leaf_nodes': None,
        'max_features': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
    }
    etc_param = {
        # class weight options
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        'presort': False,
        'random_state': None,
    }


class skRandomForest(BaseWrapperClf, _skRandomForestClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 warm_start=False, class_weight=None):
        _skRandomForestClassifier.__init__(
            self, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start, class_weight)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'n_estimators': [16, 32, 64],
        'max_depth': [i for i in range(1, 10, 2)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }

    etc_param = {
        # class weight option
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'n_jobs': 1,
        'oob_score': False,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }
    default_only_params = {
        'max_features': 'auto',
        'criterion': 'gini',
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'max_leaf_nodes': None,
        'bootstrap': True,
    }


class skExtraTrees(BaseWrapperClf, _skExtraTreesClassifier, metaclass=meta_BaseWrapperClf_with_ABC):

    def __init__(self, n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 warm_start=False, class_weight=None):
        _skExtraTreesClassifier.__init__(
            self, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start, class_weight)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'n_estimators': [16, 32, 64],
        'max_depth': [i for i in range(1, 10, 2)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    only_default_params = {
        'bootstrap': False,
        'criterion': 'gini',
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
    }
    etc_param = {
        # class weight options
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'n_jobs': 1,
        'oob_score': False,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }


class skAdaBoost(BaseWrapperClf, _skAdaBoostClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1., algorithm='SAMME.R', random_state=None):
        _skAdaBoostClassifier.__init__(self, base_estimator, n_estimators, learning_rate, algorithm, random_state)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'n_estimators': [8, 16, 32, 64, 128, 256],
        # 'base_estimator': None,
    }
    etc_param = {
        # etc
        'random_state': None,
        'algorithm': ['SAMME.R', 'SAMME'],
    }


class skGradientBoosting(BaseWrapperClf, _skGradientBoostingClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3,
                 min_impurity_decrease=0., min_impurity_split=None, init=None, random_state=None, max_features=None,
                 verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'):
        _skGradientBoostingClassifier.__init__(
            self, loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf,
            min_weight_fraction_leaf, max_depth, min_impurity_decrease, min_impurity_split, init, random_state,
            max_features, verbose, max_leaf_nodes, warm_start, presort)
        BaseWrapperClf.__init__(self)

    def _make_estimator(self, append=True):
        pass

    tuning_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_depth': [i for i in range(1, 10, 2)],
        'n_estimators': [16, 32, 64, 128, 256],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    only_default_params = {
        'criterion': 'friedman_mse',
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'max_features': None,
        'max_leaf_nodes': None,
    }
    etc_param = {
        # todo wtf?
        'init': None,
        'loss': ['deviance', 'exponential'],
        'subsample': 1.0,

        # etc
        'min_weight_fraction_leaf': 0.0,
        'presort': 'auto',
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }


class skKNeighbors(BaseWrapperClf, _skKNeighborsClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=1, **kwargs):
        _skKNeighborsClassifier.__init__(
            self, n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs, **kwargs)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'n_neighbors': [i for i in range(1, 32)],
    }
    only_default_params = {
        'weights': 'uniform',
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': 2,
        'leaf_size': 30,
    }

    etc_param = {
        'n_jobs': 1,
        'metric': 'minkowski',
        'metric_params': None,
    }


class skGaussianProcess(BaseWrapperClf, _skGaussianProcessClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, kernel=None, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, max_iter_predict=100,
                 warm_start=False, copy_X_train=True, random_state=None, multi_class="one_vs_rest", n_jobs=1):
        _skGaussianProcessClassifier.__init__(
            self, kernel, optimizer, n_restarts_optimizer, max_iter_predict, warm_start, copy_X_train, random_state,
            multi_class, n_jobs)
        BaseWrapperClf.__init__(self, )

    tuning_grid = {

    }
    remain_param = {
        'kernel': 1 ** 2 * _RBF(length_scale=1),
        'kernel__k1': 1 ** 2,
        'kernel__k1__constant_value': 1.0,
        'kernel__k1__constant_value_bounds': (1e-05, 100000.0),
        'kernel__k2': _RBF(length_scale=1),
        'kernel__k2__length_scale': 1.0,
        'kernel__k2__length_scale_bounds': (1e-05, 100000.0),
        'max_iter_predict': 100,

        'multi_class': 'one_vs_rest',
        'n_jobs': 1,
        'n_restarts_optimizer': 0,
        'optimizer': 'fmin_l_bfgs_b',
        'random_state': None,
        'warm_start': False,
        'copy_X_train': True,
    }


class skSGD(BaseWrapperClf, _skSGDClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    # todo wtf?
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                 tol=None, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=1, random_state=None,
                 learning_rate="optimal", eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False,
                 n_iter=None):
        _skSGDClassifier.__init__(
            self, loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon, n_jobs,
            random_state, learning_rate, eta0, power_t, class_weight, warm_start, average, n_iter)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        # todo random..
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 1.0],

    }

    remain_param = {
        # TODO
        'tol': None,
        'learning_rate': ['optimal', 'constant', 'invscaling'],

        'alpha': 0.0001,

        'average': False,
        'class_weight': None,
        'epsilon': 0.1,
        'eta0': 0.0,
        'fit_intercept': True,
        'l1_ratio': 0.15,
        'loss': 'hinge',
        'max_iter': None,
        'n_iter': None,

        'penalty': ['none', 'l1', 'l2', 'elasticnet'],

        'power_t': 0.5,

        # etc
        'n_jobs': 1,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'shuffle': True,
    }


class skLinear_SVC(BaseWrapperClf, _skLinearSVC, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr',
                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                 max_iter=1000):
        _skLinearSVC.__init__(
            self, penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling, class_weight, verbose,
            random_state, max_iter)

        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        'max_iter': [2 ** i for i in range(6, 13)],
    }
    only_default_params = {
        'fit_intercept': True,
        'intercept_scaling': 1,

        # todo ???
        'multi_class': ['ovr', 'crammer_singer'],
        'loss': ['squared_hinge', 'hinge'],
        'penalty': ['l2', 'l1'],
        'class_weight': None,
        'dual': True,

    }
    etc_param = {
        'random_state': None,
        'tol': 0.0001,
        'verbose': 1e-4,
    }


class skRBF_SVM(BaseWrapperClf, _skSVC, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                 tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        _skSVC.__init__(
            self, C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose,
            max_iter, decision_function_shape, random_state)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
        'C': [1 ** i for i in range(-5, 5)],
        'gamma': [1 ** i for i in range(-5, 5)],
    }
    # todo
    remain_param = {
        'cache_size': 200,
        'class_weight': None,
        'coef0': 0.0,
        'decision_function_shape': 'ovr',
        'degree': 3,
        'kernel': 'rbf',
        'max_iter': -1,
        'probability': False,
        'random_state': None,
        'shrinking': True,
        'tol': 0.001,
        'verbose': False
    }


class skVoting(BaseWrapperClf, _skVotingClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=1, flatten_transform=None):
        _skVotingClassifier.__init__(self, estimators, voting, weights, n_jobs, flatten_transform)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
    }

    tuning_params = {
    }


class skBagging(BaseWrapperClf, _BaggingClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0):
        _BaggingClassifier.__init__(
            self, base_estimator, n_estimators, max_samples, max_features, bootstrap, bootstrap_features, oob_score,
            warm_start, n_jobs, random_state, verbose)
        BaseWrapperClf.__init__(self)

    tuning_grid = {
    }

    tuning_params = {
    }
