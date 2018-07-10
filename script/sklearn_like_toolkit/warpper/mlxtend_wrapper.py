from hyperopt import hp
from mlxtend.regressor import StackingCVRegressor as _StackingCVRegressor
from mlxtend.regressor import StackingRegressor as _StackingRegressor
from mlxtend.classifier import Adaline as _Adaline
from mlxtend.classifier import EnsembleVoteClassifier as _EnsembleVoteClassifier
from mlxtend.classifier import LogisticRegression as _LogisticRegression
from mlxtend.classifier import MultiLayerPerceptron as _MultiLayerPerceptron
from mlxtend.classifier import Perceptron as _Perceptron
from mlxtend.classifier import SoftmaxRegression as _SoftmaxRegression
from mlxtend.classifier import StackingCVClassifier as _StackingCVClassifier
from mlxtend.classifier import StackingClassifier as _StackingClassifier
from mlxtend.regressor.linear_regression import \
    LinearRegression as _LinearRegression
from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.skClf_wrapper.skBernoulli_NBClf import \
    skBernoulli_NBClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperReg_with_ABC
import warnings


class mlxAdalineClf(_Adaline, BaseWrapperClf,
                    metaclass=meta_BaseWrapperClf_with_ABC):
    HyperOpt_space = {
        'eta': hp.uniform('eta', 0, 1),
        'epochs': hp.qloguniform('epochs', 3, 6, 1),
        'minibatches': hp.randint('minibatches', 16)
    }

    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'minibatches': [1, 2, 4, 8],
    }

    def __init__(self, eta=0.01, epochs=50, minibatches=1, random_seed=None,
                 print_progress=0):
        epochs = int(epochs)

        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)
        # minibatches = 1
        _Adaline.__init__(
            self, eta, epochs, minibatches, random_seed, print_progress)
        BaseWrapperClf.__init__(self)


class mlxLogisticRegressionClf(_LogisticRegression, BaseWrapperClf,
                               metaclass=meta_BaseWrapperClf_with_ABC):
    HyperOpt_space = {
        'eta': hp.uniform('eta', 0, 1),
        'epochs': hp.qloguniform('epochs', 3, 6, 1),
        'minibatches': hp.randint('minibatches', 16),
        'l2_lambda': hp.uniform('l2_lambda', 0, 1)
    }
    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'minibatches': [1, 2, 4, 8],
        'l2_lambda': [i / 10.0 for i in range(1, 10 + 1, 3)]
    }

    def __init__(self, eta=0.01, epochs=50, l2_lambda=0.0, minibatches=1,
                 random_seed=None, print_progress=0):
        epochs = int(epochs)

        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)
        _LogisticRegression.__init__(self, eta, epochs, l2_lambda, minibatches,
                                     random_seed, print_progress)
        BaseWrapperClf.__init__(self)


class mlxMLPClf(_MultiLayerPerceptron, BaseWrapperClf,
                metaclass=meta_BaseWrapperClf_with_ABC):
    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'hidden_layers': [[32, ], [64, ], [128], [32, 32], [64, 64], [128, 128]]
    }
    HyperOpt_space = {
        'eta': hp.uniform('eta', 0, 1),
        'epochs': hp.qloguniform('epochs', 3, 6, 1),
        'hidden_layers': hp.choice(
            'hidden_layers',
            [[32, ], [64, ], [128], [32, 32], [64, 64], [128, 128]])
    }

    def __init__(self, eta=0.5, epochs=50, hidden_layers=None, n_classes=None,
                 momentum=0.0, l1=0.0, l2=0.0,
                 dropout=1.0, decrease_const=0.0, minibatches=1,
                 random_seed=None, print_progress=0):
        epochs = int(epochs)

        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)
        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=RuntimeWarning)
        if hidden_layers is None:
            hidden_layers = [50]
        _MultiLayerPerceptron.__init__(
            self, eta, epochs, hidden_layers, n_classes, momentum, l1, l2,
            dropout, decrease_const, minibatches, random_seed, print_progress)
        BaseWrapperClf.__init__(self)


class mlxPerceptronClf(_Perceptron, BaseWrapperClf,
                       metaclass=meta_BaseWrapperClf_with_ABC):
    HyperOpt_space = {
        'eta': hp.uniform('eta', 0, 1),
        'epochs': hp.qloguniform('epochs', 3, 6, 1),

    }

    def __init__(self, eta=0.1, epochs=50, random_seed=None, print_progress=0):
        epochs = int(epochs)

        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)

        _Perceptron.__init__(self, eta, epochs, random_seed, print_progress)
        BaseWrapperClf.__init__(self)


class mlxSoftmaxRegressionClf(_SoftmaxRegression, BaseWrapperClf,
                              metaclass=meta_BaseWrapperClf_with_ABC):
    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'l2': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'minibatches': [1, 2, 4],
    }

    HyperOpt_space = {
        'eta': hp.uniform('eta', 0, 1),
        'epochs': hp.qloguniform('epochs', 3, 6, 1),
        'minibatches': hp.randint('minibatches', 16),
        'l2': hp.uniform('l2', 0, 1)
    }

    def __init__(self, eta=0.01, epochs=50, l2=0.0, minibatches=1,
                 n_classes=None, random_seed=None, print_progress=0):
        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)
        epochs = int(epochs)
        _SoftmaxRegression.__init__(
            self, eta, epochs, l2, minibatches, n_classes, random_seed,
            print_progress)
        BaseWrapperClf.__init__(self)


class mlxVotingClf(_EnsembleVoteClassifier):
    tuning_grid = {}

    HyperOpt_space = {}

    def __init__(self, clfs, voting='hard', weights=None, verbose=0,
                 refit=True):
        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)
        super().__init__(clfs, voting, weights, verbose, refit)


class mlxStackingClf(BaseWrapperClf, _StackingClassifier,
                     metaclass=meta_BaseWrapperClf):
    tuning_grid = {}

    HyperOpt_space = {}

    def __init__(
            self, classifiers, meta_classifier=None, use_probas=False,
            average_probas=False, verbose=0, use_features_in_secondary=False,
            store_train_meta_features=False, use_clones=True):
        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)
        if meta_classifier is None:
            meta_classifier = skBernoulli_NBClf()

        BaseWrapperClf.__init__(self)
        _StackingClassifier.__init__(
            self, classifiers, meta_classifier, use_probas, average_probas,
            verbose, use_features_in_secondary, store_train_meta_features,
            use_clones)

    def score_pack(self, X, y):
        y = self.np_arr_to_index(y)
        return self._apply_metric_pack(y, self.predict(X))


class mlxStackingCVClf(BaseWrapperClf, _StackingCVClassifier,
                       metaclass=meta_BaseWrapperClf):
    tuning_grid = {}

    HyperOpt_space = {}

    def __init__(self, classifiers, meta_classifier=None, use_probas=False,
                 cv=2, use_features_in_secondary=False,
                 stratify=True, shuffle=True, verbose=0,
                 store_train_meta_features=False, use_clones=True):
        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)

        if meta_classifier is None:
            meta_classifier = skBernoulli_NBClf()

        BaseWrapperClf.__init__(self)
        _StackingCVClassifier.__init__(
            self, classifiers, meta_classifier, use_probas, cv,
            use_features_in_secondary, stratify, shuffle, verbose,
            store_train_meta_features, use_clones)

    def score_pack(self, X, y):
        return self._apply_metric_pack(y, self.predict(X))


class mlxLinearReg(BaseWrapperReg, _LinearRegression,
                   metaclass=meta_BaseWrapperReg_with_ABC):
    HyperOpt_space = {
        'eta': hp.uniform('eta', 0, 1),
        'epochs': hp.qloguniform('epochs', 3, 6, 1),
        'minibatches': hp.randint('minibatches', 16)
    }

    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'minibatches': [1, 2, 4, 8],
    }

    def __init__(self, eta=0.01, epochs=50, minibatches=None, random_seed=None,
                 print_progress=0):
        epochs = int(epochs)
        _LinearRegression.__init__(
            self, eta, epochs, minibatches, random_seed, print_progress)

        BaseWrapperReg.__init__(self)

        warnings.filterwarnings(module='mlxtend*', action='ignore',
                                category=FutureWarning)

    def score(self, X, y, sample_weight=None):
        predict = self.predict(X)
        return self._apply_metric(y, predict, 'explained_variance_score')


class mlxStackingCVReg(BaseWrapperReg, _StackingCVRegressor,
                       metaclass=meta_BaseWrapperReg_with_ABC):
    tuning_grid = {}

    HyperOpt_space = {}

    def __init__(self, regressors, meta_regressor, cv=5, shuffle=True,
                 use_features_in_secondary=False,
                 store_train_meta_features=False, refit=True):
        _StackingCVRegressor.__init__(
            self, regressors, meta_regressor, cv, shuffle,
            use_features_in_secondary, store_train_meta_features, refit)
        BaseWrapperReg.__init__(self)


class mlxStackingReg(BaseWrapperReg, _StackingRegressor,
                     metaclass=meta_BaseWrapperReg_with_ABC):
    tuning_grid = {}

    HyperOpt_space = {}

    def __init__(self, regressors, meta_regressor, verbose=0,
                 store_train_meta_features=False, refit=True):
        _StackingRegressor.__init__(
            self, regressors, meta_regressor, verbose,
            store_train_meta_features, refit)
        BaseWrapperReg.__init__(self)
