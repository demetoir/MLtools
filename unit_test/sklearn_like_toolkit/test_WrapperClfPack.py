from pprint import pprint
from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skMLP
from script.sklearn_like_toolkit.wrapperGridSearchCV import wrapperGridSearchCV
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack

datapack = DatasetPackLoader().load_dataset("titanic")
train_set = datapack.pack['train']
train_set, valid_set = train_set.split('train', 'train', 'valid', (7, 3))
train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])


# print('reset current dir')
#
# print('cur dir')
# print(os.getcwd())
# head, tail = os.path.split(os.getcwd())
# os.chdir(head)
# print(os.getcwd())


def test_save_params():
    clf_pack = ClassifierPack(['skMLP'])
    clf_pack.param_search(train_Xs, train_Ys)
    train_score = clf_pack.score(train_Xs, train_Ys)
    valid_score = clf_pack.score(valid_Xs, valid_Ys)

    pprint(train_score)
    pprint(valid_score)


def test_load_params():
    clf_pack = ClassifierPack(['skMLP'])
    clf_pack.param_search(train_Xs, train_Ys)
    train_score = clf_pack.score(train_Xs, train_Ys)
    valid_score = clf_pack.score(valid_Xs, valid_Ys)

    pprint(train_score)
    pprint(valid_score)

    path = clf_pack.save_params()
    clf_pack.load_params(path)
    clf_pack.fit(train_Xs, train_Ys)
    train_score = clf_pack.score(train_Xs, train_Ys)
    valid_score = clf_pack.score(valid_Xs, valid_Ys)
    pprint(train_score)
    pprint(valid_score)


def test_hard_voting():
    clf = ClassifierPack()
    clf = clf.make_FoldingHardVote()
    clf.fit(train_Xs, train_Ys)

    predict = clf.predict(valid_Xs[:4])
    print(f'predict {predict}')

    proba = clf.predict_proba(valid_Xs[:4])
    print(f'proba {proba}')

    predict_bincount = clf.predict_bincount(valid_Xs[:4])
    print(f'predict_bincount {predict_bincount}')

    score = clf.score(valid_Xs, valid_Ys)
    print(f'score {score}')


def test_stacking(self):
    ClassifierPack = self.cls
    train_Xs = self.train_Xs
    train_Ys = self.train_Ys
    valid_Xs = self.valid_Xs
    valid_Ys = self.valid_Ys

    clf = ClassifierPack()
    metaclf = clf.pack['XGBoost']
    clf = clf.make_stackingClf(metaclf)
    clf.fit(train_Xs, train_Ys)

    predict = clf.predict(valid_Xs[:4])
    print(f'predict {predict}')

    proba = clf.predict_proba(valid_Xs[:4])
    print(f'proba {proba}')

    score = clf.score(valid_Xs, valid_Ys)
    print(f'score {score}')
    pass


def test_stackingCV():
    clf = ClassifierPack()
    metaclf = clf.pack['XGBoost']
    clf = clf.make_stackingCVClf(metaclf)
    clf.fit(train_Xs, train_Ys)

    predict = clf.predict(valid_Xs[:4])
    print(f'predict {predict}')

    proba = clf.predict_proba(valid_Xs[:4])
    print(f'proba {proba}')

    score = clf.score(valid_Xs, valid_Ys)
    print(f'score {score}')


def test_clf_pack_param_search():
    clf = ClassifierPack()
    clf.param_search(train_Xs, train_Xs)

    pprint('train score', clf.score(train_Xs, train_Ys))
    pprint('test score', clf.score(valid_Xs, valid_Ys))
    pprint('predict', clf.predict(valid_Xs[:2]))
    pprint('predict_proba', clf.predict_proba(valid_Xs[:2]))


def test_clfpack():
    clf = ClassifierPack()
    clf.fit(train_Xs, train_Ys)
    predict = clf.predict(valid_Xs[:2])
    pprint('predict', predict)
    proba = clf.predict_proba(valid_Xs[:2])
    pprint('predict_proba', proba)

    score = clf.score(valid_Xs, valid_Ys)
    pprint('test score', score)

    score_pack = clf.score_pack(valid_Xs, valid_Ys)
    pprint('score pack', score_pack)

    # initialize data


def test_pickle_clf_pack():
    clf_pack = ClassifierPack()
    clf_pack.fit(train_Xs, train_Ys)

    score = clf_pack.score_pack(train_Xs, train_Ys)
    pprint(score)
    clf_pack.dump('./clf.pkl')

    clf_pack = clf_pack.load('./clf.pkl')
    score = clf_pack.score_pack(train_Xs, train_Ys)
    pprint(score)


def test_wrapperGridSearchCV():
    clf_cls = skMLP
    base_clf = clf_cls()
    param_search = wrapperGridSearchCV(base_clf, clf_cls.tuning_grid)

    param_search.fit(train_Xs, train_Ys)
    result = param_search.cv_results_
    pprint(result)

    best_clf = param_search.best_estimator_
    pprint(best_clf)

    score = best_clf.score(valid_Xs, valid_Ys)
    pprint(score)

    score = best_clf.score_pack(valid_Xs, valid_Ys)
    pprint(score)


def test_wrapper_pack_grid_search():
    path = './test_wrapper_pack_grid_search.pkl'
    clf_pack = ClassifierPack()
    clf_pack.fit(train_Xs, train_Ys)
    clf_pack.gridSearchCV(train_Xs, train_Ys)
    score = clf_pack.score_pack(train_Xs, train_Ys)
    pprint(score)
    clf_pack.dump(path)

    clf_pack = ClassifierPack().load(path)
    score = clf_pack.score_pack(train_Xs, train_Ys)
    pprint(score)
    score = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint(score)

    score = clf_pack.score(valid_Xs, valid_Ys)
    pprint(score)
    result = clf_pack.optimize_result
    pprint(result)
