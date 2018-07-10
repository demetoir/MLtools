from pprint import pprint
from hyperopt import hp
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.sklearn_like_toolkit.HyperOpt.FreeTrials import FreeTrials
from script.sklearn_like_toolkit.HyperOpt.HyperOpt import HyperOpt, HyperOpt_fn
from script.util.deco import deco_timeit

data_pack = DatasetPackLoader().load_dataset('titanic')


def fit_clf(params):
    train_set = data_pack['train']
    train_set.shuffle()
    train_set, valid_set = train_set.split()
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf_pack = ClassifierPack()
    clf = clf_pack['skMLPClf']
    clf = clf.__class__(**params)
    clf.fit(train_Xs, train_Ys)
    score = -clf.score(valid_Xs, valid_Ys)

    return score


@deco_timeit
def test_hyperOpt():
    data_pack = DatasetPackLoader().load_dataset('titanic')

    def fit_clf(kwargs):
        params = kwargs['params']
        data_pack = kwargs['data_pack']
        train_set = data_pack['train']
        train_set.shuffle()
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
    best = opt.fit_serial(fit_clf, data_pack, space, 10)
    pprint(best)
    pprint(opt.trials)
    pprint(opt.losses)
    pprint(opt.result)
    pprint(opt.opt_info)
    pprint(opt.best_param)
    pprint(opt.best_loss)
    pprint(opt.statuses)


@deco_timeit
def test_HyperOpt_parallel():
    clf_pack = ClassifierPack()
    clf = clf_pack['skMLPClf']
    space = clf.HyperOpt_space

    opt = HyperOpt()
    trials1 = opt.fit_serial(fit_clf, space, 1, trials=FreeTrials())
    # pprint(trials)
    # pprint(opt.trials)
    # pprint(opt.losses)
    # pprint(opt.result)
    # pprint(opt.opt_info)
    pprint(opt.best_param)
    pprint(opt.best_loss)
    pprint(len(trials1))
    # pprint(len(opt.outer_trials))

    trials2 = opt.fit_serial(fit_clf, space, 2, trials=FreeTrials())
    pprint(opt.best_param)
    pprint(opt.best_loss)
    pprint(len(trials2))

    # pprint(trials1.trials)
    # pprint(trials2.trials)
    # pprint(trials1.__dict__['_dynamic_trials'])
    # pprint(trials2.__dict__['_dynamic_trials'])

    trials3 = trials1.concat(trials2)
    # pprint(trials3.__dict__['_dynamic_trials'])

    pprint(len(trials3))

    trials3 = opt.fit_serial(fit_clf, space, 1, trials=trials3)
    pprint(opt.best_param)
    pprint(opt.best_loss)
    # pprint(trials3.__dict__['_dynamic_trials'])
    pprint(len(trials3))

    trials4 = opt.fit_serial(fit_clf, space, 2, trials=FreeTrials.deepcopy(trials3))
    pprint(opt.best_param)
    pprint(opt.best_loss)
    # pprint(trials3.__dict__['_dynamic_trials'])
    pprint(len(trials4))

    trials5 = opt.fit_serial(fit_clf, space, 3, trials=FreeTrials.deepcopy(trials3))
    pprint(opt.best_param)
    pprint(opt.best_loss)
    # pprint(trials3.__dict__['_dynamic_trials'])
    pprint(len(trials5))

    trials6 = trials4.concat(trials5)
    pprint(len(trials6))

    trials6 = opt.fit_serial(fit_clf, space, 2, trials=trials6)
    pprint(opt.best_param)
    pprint(opt.best_loss)
    # pprint(trials3.__dict__['_dynamic_trials'])
    pprint(len(trials6))

    partial = trials6[:5]
    partial = FreeTrials.partial_deepcopy(trials6, 0, 5)
    pprint(partial)
    pprint(len(partial))

    partial = opt.fit_serial(fit_clf, space, 4, trials=partial)
    pprint(opt.best_param)
    pprint(opt.best_loss)
    # pprint(trials3.__dict__['_dynamic_trials'])

    partial = opt.fit_serial(fit_clf, space, 30, trials=FreeTrials())
    pprint(opt.best_param)
    pprint(opt.best_loss)

    n = 100
    partial = opt.fit_parallel(fit_clf, space, n, trials=FreeTrials())
    # pprint(opt.best_param)
    pprint(opt.best_loss)
    pprint(len(opt.trials))
    # pprint(opt.trials)


class fit_fn(HyperOpt_fn):

    @staticmethod
    def fn(params, feed_args, feed_kwargs):
        # pprint(params)
        # pprint(feed_args)
        # pprint(feed_kwargs)

        data_pack = feed_kwargs['data_pack']
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


def test_HyperOpt_space_with_data():
    data_pack = DatasetPackLoader().load_dataset('titanic')
    clf_name = 'skMLPClf'
    clf_pack = ClassifierPack()

    train_set = data_pack['train']

    # train_set, valid_set = train_set.split()
    # train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    # valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])
    # clf_pack.fit(train_Xs, train_Ys)
    # score = clf_pack.score(valid_Xs, valid_Ys)
    # pprint(score)

    clf = clf_pack[clf_name]
    space = clf.HyperOpt_space

    # data_pack = None
    opt = HyperOpt()

    # space.update({'args': ()})
    # space.update({'kwargs':})
    opt.fit_serial(fit_fn, space, 10, feed_args=(), feed_kwargs={'data_pack': data_pack.to_DummyDatasetPack()})
    pprint(opt.best_loss)
    pprint(opt.best_param)

    opt.fit_parallel(fit_fn, space, 10, feed_args=(), feed_kwargs={'data_pack': data_pack.to_DummyDatasetPack()})
    pprint(opt.best_loss)
    pprint(opt.best_param)
