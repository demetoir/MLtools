from hyperopt import hp
from sklearn.neural_network import MLPRegressor as _MLPRegressor

from script.sklearn_like_toolkit.warpper.base.BaseWrapperReg import BaseWrapperReg
from script.sklearn_like_toolkit.warpper.base.MixIn import MetaBaseWrapperRegWithABC


class skMLPReg(_MLPRegressor, BaseWrapperReg, metaclass=MetaBaseWrapperRegWithABC):

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
                 random_state=None, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        _MLPRegressor.__init__(
            self, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init,
            power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum,
            nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon)
        BaseWrapperReg.__init__(self)

    HyperOpt_space = {
        'alpha': hp.loguniform('alpha', -4, 3),
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'activation': hp.choice('activation', ['relu']),
        # 'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100,), (100, 100,)]),
        'solver': 'adam',
        'beta_1': hp.loguniform('beta_1', -1, 0),
        'beta_2': hp.loguniform('beta_2', -1, 0),
        # 'epsilon': 1e-08,

        'tol': 0.0001,
        'max_iter': 1000,
        'early_stopping': hp.choice('early_stopping', [True, False]),
        'batch_size': 'auto',
        'shuffle': True,
        'validation_fraction': 0.1,
    }

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
