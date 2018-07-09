from pprint import pprint
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.sklearn_like_toolkit.warpper.wrapperRandomizedSearchCV import wrapperRandomizedSearchCV
from script.util.deco import deco_timeit


@ deco_timeit
def test_wrapperRandomizedSearchCV():
    data_pack = DatasetPackLoader().load_dataset('titanic')
    train_set = data_pack['train']

    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])

    clf_pack = ClassifierPack()
    clf = clf_pack['skMLPClf']

    from scipy import stats

    tuning_distributions = {
        'activation': ['relu'],
        'alpha': stats.expon(scale=0.001),
        # 'hidden_layer_sizes': [(128, 128), (64, 64), ],

        'max_iter': [1000],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        # 'learning_rate_init': stats.expon(scale=0.001),
        # 'tol': [0.0001],
    }
    # 'alpha': hp.loguniform('alpha', -4, 3),
    # 'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),


    param_distributions = tuning_distributions
    search = wrapperRandomizedSearchCV(clf, param_distributions, n_iter=10)
    search.fit(train_Xs, train_Ys)
    # pprint(search.best_estimator_)
    # pprint(search.best_params_)
    # pprint(search.best_index_)
    pprint(search.best_score_)
    # pprint(search.cv_results_)

    pass