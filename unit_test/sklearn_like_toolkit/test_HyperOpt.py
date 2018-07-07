from pprint import pprint
from hyperopt import hp
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.sklearn_like_toolkit.HyperOpt import HyperOpt
from script.util.deco import deco_timeit


@deco_timeit
def test_hyperOpt():
    data_pack = DatasetPackLoader().load_dataset('titanic')

    def fit_clf(kwargs):
        params = kwargs['params']
        data_pack = kwargs['data_pack']
        train_set = data_pack['train']
        train_set, valid_set = train_set.split()
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

        clf_pack = ClassifierPack()
        clf = clf_pack['skMLPClf']
        clf = clf.__class__(**params)
        clf.fit(train_Xs, train_Ys)
        score = -clf.score(valid_Xs, valid_Ys)

        return score

    space = hp.choice('classifier_type', [{
        'alpha': hp.loguniform('alpha', -4, 3),
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'activation': hp.choice('activation', ['relu']),
        'max_iter': 1000,
    }])

    opt = HyperOpt()
    best = opt.fit(fit_clf, data_pack, space, 30)
    pprint(best)
    pprint(opt.trials)
    pprint(opt.losses)
    pprint(opt.result)
    pprint(opt.opt_info)
    pprint(opt.best_param)
    pprint(opt.best_loss)
    pprint(opt.statuses)
