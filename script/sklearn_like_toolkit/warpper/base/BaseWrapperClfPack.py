import numpy as np
from tqdm import tqdm
from script.data_handler.DFEncoder import DFEncoder
from script.sklearn_like_toolkit.warpper.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.warpper.base.MixIn import ClfWrapperMixIn, MetaBaseWrapperClf
from script.util.misc_util import log_error_trace


class BaseWrapperClfPack(ClfWrapperMixIn, metaclass=MetaBaseWrapperClf):
    class_pack = {}

    @staticmethod
    def _clone(clf):

        return super()._clone(clf)

    def __init__(self, pack_keys=None, n_classes=None, x_df_encoder=None, y_df_encoder=None):
        super().__init__()

        if pack_keys is None:
            pack_keys = self.class_pack.keys()

        self.pack = {}
        for key in pack_keys:
            self.pack[key] = self.class_pack[key]()

        self.n_classes = n_classes

        if x_df_encoder:
            self.x_df_encoder = x_df_encoder
        else:
            self.x_df_encoder = DFEncoder()

        if y_df_encoder:
            self.y_df_encoder = y_df_encoder
        else:
            self.y_df_encoder = DFEncoder()

        self.is_encoded = False

    def __str__(self):
        return self.__class__.__name__

    def __getitem__(self, item) -> BaseWrapperClf:
        return self.pack.__getitem__(item)

    @property
    def feature_importance(self):
        ret = {}
        for key, clf in tqdm(self.pack.items()):
            tqdm.write(f'collect feature importance {key}')

            if hasattr(clf, 'feature_importances_'):
                ret[key] = getattr(clf, 'feature_importances_')
        return ret

    def _check_n_class(self, y):
        y_onehot = self.np_arr_to_onehot(y)
        if self.n_classes is None:
            if y_onehot.ndim == 1:
                self.n_classes = y_onehot.shape[0]
            elif y_onehot.ndim == 2:
                self.n_classes = y_onehot.shape[1]
            else:
                raise TypeError(f'y expect shape (-1) or (-1,-1), but {y_onehot.shape}')

        y_label = self.np_arr_to_index(y)

        feed_n_classes = np.unique(y_label, axis=0).shape[0]
        if feed_n_classes != self.n_classes:
            raise TypeError(f'expect {self.n_classes} class, but only {feed_n_classes} class')

    def fit(self, x, y):
        x, y = self._if_df_encode(x, y)
        label_Ys = self.np_arr_to_index(y)

        self._check_n_class(y)

        for key in tqdm(self.pack):
            tqdm.write(f'fit {key}')

            try:
                self.pack[key].fit(x, label_Ys)
            except BaseException as e:
                log_error_trace(self.log.warn, e)
                self.log.warn(f'while fit, {key} raise {e}')

    def predict(self, x, decode_df=True):
        x = self._if_df_encode_x(x)

        result = {}
        for key in tqdm(self.pack):
            tqdm.write(f'predict {key}')

            try:
                predict = self.pack[key].predict(x)

                if decode_df:
                    predict = self.np_arr_to_onehot(predict, n=self.n_classes)
                    predict = self._if_df_decode_y(predict)

                result[key] = predict
            except BaseException as e:
                self.log.warn(f'while predict, {key} raise {e}')

        return result

    def predict_proba(self, x):
        x = self._if_df_encode_x(x)

        result = {}
        for key in tqdm(self.pack):
            tqdm.write(f'predict_proba {key}')

            try:
                result[key] = self.pack[key].predict_proba(x)
            except BaseException as e:
                self.log.warn(f'while predict_proba, {key} raise {e}')
        return result

    def predict_confidence(self, x):
        x = self._if_df_encode_x(x)

        confidences = {}
        for key, clf in tqdm(self.pack.items()):
            tqdm.write(f'predict_confidence {key}')

            try:
                confidences[key] = clf.predict_confidence(x)
            except BaseException as e:
                log_error_trace(self.log.warn, e,
                                f'while execute confidence at {key},\n')

        return confidences

    def score(self, x, y, metric='accuracy'):
        x, y = self._if_df_encode(x, y)
        y = self.np_arr_to_index(y)

        scores = {}
        for clf_k, predict in tqdm(self.predict(x, decode_df=False).items()):
            tqdm.write(f'score {clf_k}')

            scores[clf_k] = self._apply_metric(y, predict, metric)
        return scores

    def score_pack(self, x, y):
        x, y = self._if_df_encode(x, y)
        y = self.np_arr_to_index(y)

        ret = {}
        for clf_k, predict in tqdm(self.predict(x, decode_df=False).items()):
            tqdm.write(f'score_pack {clf_k}')

            ret[clf_k] = self._apply_metric_pack(y, predict)
        return ret

    def to_pickle(self, path, **kwargs):
        self.log.info(f'save pickle at {path}')
        super().to_pickle(path)

    def from_pickle(self, path, overwrite_self=False, **kwargs):
        self.log.info(f'load pickle at {path}')
        return super().from_pickle(path, overwrite_self)

    def drop(self, key):
        self.pack.pop(key)
