from abc import ABCMeta
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from util.MixIn import PickleMixIn, LoggerMixIn
from util.misc_util import log_error_trace
from util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_INDEX, NP_ARRAY_TYPE_ONEHOT


class meta_BaseWrapperClf(type):
    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        def deco_reformat_y(func):
            def wrapper(*args, **kwargs):
                y = args[2]
                args = list(args[:2]) + [Reformat_Ys_MixIn.np_arr_to_index(y)] + list(args[3:])

                ret = func(*args, **kwargs)
                return ret

            return wrapper

        func_names = ['score', 'score_pack', 'fit']
        for func_name in func_names:
            if hasattr(cls, func_name):
                func = getattr(cls, func_name)
                setattr(cls, func_name, deco_reformat_y(func))


class meta_BaseWrapperClf_with_ABC(meta_BaseWrapperClf, ABCMeta):
    pass


class Reformat_Ys_MixIn:
    @staticmethod
    def np_arr_to_index(Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_INDEX)

    @staticmethod
    def np_arr_to_onehot(Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_ONEHOT)


CLF_METRICS = {
    'accuracy': accuracy_score,
    'confusion_matrix': confusion_matrix,
    'roc_auc_score': roc_auc_score,
    'recall_score': recall_score,
    'precision_score': precision_score,
}


class score_pack_MixIn(Reformat_Ys_MixIn):
    def __init__(self):
        Reformat_Ys_MixIn.__init__(self)
        self._metrics = CLF_METRICS

    def _apply_metric(self, Y_true, Y_predict, metric):
        return self._metrics[metric](Y_true, Y_predict)

    def _apply_metric_pack(self, Y_true, Y_predict):
        ret = {}
        for key in self._metrics:
            try:
                ret[key] = self._apply_metric(Y_true, Y_predict, key)
            except BaseException as e:
                log_error_trace(getattr(self, 'log').warn, e,
                                head=f'while {self.__class__} execute score_pack, skip to applying metric {key}\n')
        return ret

    def score_pack(self, X, y):
        y = self.np_arr_to_index(y)
        return self._apply_metric_pack(y, getattr(self, 'predict')(X))


class DummyParamMixIN:
    def get_params(self, deep=True):
        if hasattr(self, 'get_params'):
            return getattr(self, 'get_params')(self, deep=deep)
        else:
            return {}

    def set_params(self, **params):
        if hasattr(self, 'get_params'):
            return getattr(self, 'get_params')(self, **params)
        else:
            return None


class ClfConfidenceMixIn:
    @staticmethod
    def _apply_confidence(proba):
        shape = proba.shape
        batch_size = shape[0]
        n_class = shape[1]

        np_arr = np.abs(1.0 / n_class - proba)
        np_arr = np_arr.sum(axis=1)
        return np_arr

    def _predict_confidence(self, Xs):
        if hasattr(self, 'predict_proba'):
            func = getattr(self, 'predict_proba')
            confidences = func(Xs)
        else:
            getattr(self, 'log').warn(f'skip predict_confidence, {self.__class__} has no predict_proba')
            confidences = None

        return confidences

    def predict_confidence(self, Xs):
        return self._predict_confidence(Xs)


class etc_MixIn:
    @staticmethod
    def _clone(clf):
        return clf.__class__(**clf.get_params())


class ClfWrapperMixIn(score_pack_MixIn, PickleMixIn, LoggerMixIn, DummyParamMixIN, ClfConfidenceMixIn, etc_MixIn):
    def __init__(self):
        score_pack_MixIn.__init__(self)
        PickleMixIn.__init__(self)
        LoggerMixIn.__init__(self)
        DummyParamMixIN.__init__(self)
        ClfConfidenceMixIn.__init__(self)
        etc_MixIn.__init__(self)
