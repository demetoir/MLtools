from sklearn_like_toolkit.base.MixIn import Reformat_Ys_MixIn, clf_metric_MixIn
from util.MixIn import PickleMixIn, LoggerMixIn


class BaseWrapperClf(Reformat_Ys_MixIn, clf_metric_MixIn, LoggerMixIn, PickleMixIn):
    @staticmethod
    def _clone(clf):
        return clf.__class__(**clf.get_params())

    def fit(self, Xs, Ys):
        raise NotImplementedError

    def predict_proba(self, Xs):
        raise NotImplementedError

    def predict(self, Xs):
        raise NotImplementedError

    def score(self, Xs, Ys, metric='accuracy'):
        raise NotImplementedError

    def score_pack(self, Xs, Ys):
        raise NotImplementedError
