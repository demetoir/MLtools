from script.sklearn_like_toolkit.ClassifierPack import ClassifierPack
from script.sklearn_like_toolkit.param_search.ParamOpt import ParamOpt


def test_ParamOpt():
    clf_pack = ClassifierPack(['skDecisionTreeClf'])
    dtree = clf_pack['skDecisionTreeClf']

    from sklearn.datasets import load_iris
    data = load_iris()
    y = data.target
    x = data.data

    opt = ParamOpt(cv=10, n_iter=10, n_jobs=1)
    dtree = opt.fit(dtree, x, y)

    train_score = dtree.score(x, y)
    print(train_score)

    clf_pack = ClassifierPack(['skDecisionTreeClf'])
    dtree = clf_pack['skDecisionTreeClf']

    from sklearn.datasets import load_iris
    data = load_iris()
    y = data.target
    x = data.data

    opt = ParamOpt(cv=10, n_iter=10, n_jobs=2)
    dtree = opt.fit(dtree, x, y)

    train_score = dtree.score(x, y)
    print(train_score)
