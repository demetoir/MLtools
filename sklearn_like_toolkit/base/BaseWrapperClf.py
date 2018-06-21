from sklearn_like_toolkit.base.MixIn import Reformat_Ys_MixIn, clf_metric_MixIn
from util.MixIn import PickleMixIn, LoggerMixIn
import numpy as np

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

    def _apply_confidence(self, proba):
        shape = proba.shape
        batch_size = shape[0]
        n_class = shape[1]

        np_arr = np.abs(1.0 / n_class - proba)
        np_arr = np_arr.sum(axis=1)
        return np_arr

    def predict_confidence(self, Xs):
        confidences = {}
        for clf_k, proba in self.predict_proba(Xs).items():
            confidences[clf_k] = self._apply_confidence(proba)

        return confidences
