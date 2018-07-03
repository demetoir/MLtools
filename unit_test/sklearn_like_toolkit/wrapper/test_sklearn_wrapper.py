from script.sklearn_like_toolkit.warpper.sklearn_wrapper import skRidgeClf
from unit_test.sklearn_like_toolkit.wrapper.helper import wrapper_clf_common


def test_skRidgeClf():
    clf = skRidgeClf()
    wrapper_clf_common(clf)
