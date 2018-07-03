from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skRidgeClf, skRidgeCVClf, skPassiveAggressiveClf, \
    skRadiusNeighborsClf, skKNeighborsClf, skNearestCentroidClf
from unit_test.sklearn_like_toolkit.wrapper.helper import wrapper_clf_common


def test_skRidgeClf():
    clf = skRidgeClf()
    wrapper_clf_common(clf)


def test_skRidgeCVClf():
    clf = skRidgeCVClf()
    wrapper_clf_common(clf)


def test_skPassiveAggressiveClf():
    clf = skPassiveAggressiveClf()
    wrapper_clf_common(clf)


def test_skRadiusNeighborsClf():
    clf = skRadiusNeighborsClf()
    wrapper_clf_common(clf)


def test_skKNeighborsClf():
    clf = skKNeighborsClf()
    wrapper_clf_common(clf)


def test_skNearestCentroidClf():
    clf = skNearestCentroidClf()
    wrapper_clf_common(clf)
