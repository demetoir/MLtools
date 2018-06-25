from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skBernoulli_NB
from script.sklearn_like_toolkit.wrapperGridSearchCV import wrapperGridSearchCV
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack

meta_clf = skBernoulli_NB()
datapack = DatasetPackLoader().load_dataset("titanic")
train_set = datapack['train']
train_set, valid_set = train_set.split((7, 3))
train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])
sample_Xs, sample_Ys = valid_Xs[:2], valid_Ys[:2]


def test_param_search():
    clf_pack = ClassifierPack(['skMLP'])
    clf_pack.param_search(train_Xs, train_Ys)
    train_score = clf_pack.score(train_Xs, train_Ys)
    valid_score = clf_pack.score(valid_Xs, valid_Ys)
    print(train_score)
    print(valid_score)

    path = clf_pack.save_params()
    clf_pack.load_params(path)
    clf_pack.fit(train_Xs, train_Ys)
    train_score = clf_pack.score(train_Xs, train_Ys)
    valid_score = clf_pack.score(valid_Xs, valid_Ys)
    print(train_score)
    print(valid_score)


def test_make_FoldingHardVote():
    clf = ClassifierPack(['skMLP', "skBernoulli_NB", "skDecisionTree"])
    clf = clf.make_FoldingHardVote()
    clf.fit(train_Xs, train_Ys)

    predict = clf.predict(sample_Xs)
    print(f'predict {predict}')

    proba = clf.predict_proba(sample_Xs)
    print(f'proba {proba}')

    predict_bincount = clf.predict_bincount(sample_Xs)
    print(f'predict_bincount {predict_bincount}')

    score = clf.score(sample_Xs, sample_Ys)
    print(f'score {score}')


def test_make_stackingClf():
    clf = ClassifierPack(['skMLP', "skBernoulli_NB", "skDecisionTree"])

    clf = clf.make_stackingClf(meta_clf)
    clf.fit(train_Xs, train_Ys)

    predict = clf.predict(valid_Xs[:4])
    print(f'predict {predict}')

    proba = clf.predict_proba(valid_Xs[:4])
    print(f'proba {proba}')

    score = clf.score(valid_Xs, valid_Ys)
    print(f'score {score}')


def test_make_stackingCVClf():
    clf = ClassifierPack(['skMLP', "skBernoulli_NB", "skDecisionTree"])
    meta_clf = clf.pack["skBernoulli_NB"]
    clf = clf.make_stackingCVClf(meta_clf)
    clf.fit(train_Xs, train_Ys)

    predict = clf.predict(valid_Xs[:4])
    print(f'predict {predict}')

    proba = clf.predict_proba(valid_Xs[:4])
    print(f'proba {proba}')

    score = clf.score(valid_Xs, valid_Ys)
    print(f'score {score}')


def test_ClassifierPack():
    clf = ClassifierPack(['skMLP', "skBernoulli_NB", "skDecisionTree"])
    clf.fit(train_Xs, train_Ys)
    predict = clf.predict(valid_Xs[:2])
    print('predict', predict)
    proba = clf.predict_proba(valid_Xs[:2])
    print('predict_proba', proba)

    score = clf.score(valid_Xs, valid_Ys)
    print('test score', score)

    score_pack = clf.score_pack(valid_Xs, valid_Ys)
    print('score pack', score_pack)


def test_pickle_clf_pack():
    clf_pack = ClassifierPack(['skMLP'])
    clf_pack.fit(train_Xs, train_Ys)

    score = clf_pack.score_pack(train_Xs, train_Ys)
    print(score)
    clf_pack.dump('./test_pickle_clf_pack.pkl')

    clf_pack = clf_pack.load('./test_pickle_clf_pack.pkl')
    score = clf_pack.score_pack(train_Xs, train_Ys)
    print(score)


def test_wrapperGridSearchCV():
    clf_cls = skBernoulli_NB
    base_clf = clf_cls()
    param_search = wrapperGridSearchCV(base_clf, clf_cls.tuning_grid)

    param_search.fit(train_Xs, train_Ys)
    result = param_search.cv_results_
    print(result)

    best_clf = param_search.best_estimator_
    print(best_clf)

    score = best_clf.score(valid_Xs, valid_Ys)
    print(score)

    score = best_clf.score_pack(valid_Xs, valid_Ys)
    print(score)


def test_wrapper_pack_grid_search():
    path = './test_wrapper_pack_grid_search.pkl'
    clf_pack = ClassifierPack(['skMLP', "skBernoulli_NB", "skDecisionTree"])
    clf_pack.fit(train_Xs, train_Ys)
    clf_pack.gridSearchCV(train_Xs, train_Ys)
    score = clf_pack.score_pack(train_Xs, train_Ys)
    print(score)
    clf_pack.dump(path)

    clf_pack = ClassifierPack().load(path)
    score = clf_pack.score_pack(train_Xs, train_Ys)
    print(score)
    score = clf_pack.score_pack(valid_Xs, valid_Ys)
    print(score)

    score = clf_pack.score(valid_Xs, valid_Ys)
    print(score)
    result = clf_pack.optimize_result
    print(result)
