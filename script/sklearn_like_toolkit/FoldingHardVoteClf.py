import numpy as np
from script.data_handler.DummyDataset import DummyDataset
from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf


class FoldingHardVoteClf(BaseWrapperClf):
    def __init__(self, clfs, split_rate=0.8):
        super().__init__()
        self.clfs = [self._clone(clf) for clf in clfs]
        self.n = len(self.clfs)
        self.class_size = None
        self._is_fitted = False
        self.split_rate = split_rate

    def _collect_predict(self, Xs):
        return np.array([clf.predict(Xs) for clf in self.clfs])

    def fit(self, Xs, Ys, **kwargs):
        self.class_size = self.np_arr_to_onehot(Ys).shape[1]
        dset = DummyDataset()
        dset.add_data('Xs', Xs)
        dset.add_data('Ys', Ys)

        for clf in self.clfs:
            dset.shuffle()
            Xs, Ys = dset.next_batch(int(dset.size * self.split_rate))
            clf.fit(Xs, Ys)

        self._is_fitted = True

    def predict_bincount(self, Xs):
        predicts = self._collect_predict(Xs).transpose()
        predicts = predicts.astype(np.int64)
        bincount = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.class_size),
            axis=1, arr=predicts
        )
        return bincount

    def predict_proba(self, Xs, **kwargs):
        return self.predict_bincount(Xs) / float(self.n)

    def predict(self, Xs, **kwargs):
        predicts = self._collect_predict(Xs).transpose()
        predicts = predicts.astype(int)
        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, minlength=self.class_size)),
            axis=1, arr=predicts)
        return maj

    def score(self, Xs, Ys, metric='accuracy'):
        Ys = self.np_arr_to_index(Ys)
        return self._apply_metric(Ys, self.predict(Xs), metric)

    def score_pack(self, Xs, Ys):
        Ys = self.np_arr_to_index(Ys)
        return self._apply_metric_pack(Ys, self.predict(Xs))
