from abc import ABCMeta
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from script.util.MixIn import PickleMixIn, LoggerMixIn
from script.util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_INDEX, NP_ARRAY_TYPE_ONEHOT
from sklearn.metrics.regression import r2_score
from sklearn.metrics.regression import explained_variance_score
from sklearn.metrics.regression import mean_absolute_error
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics.regression import mean_squared_log_error
from sklearn.metrics.regression import median_absolute_error


class meta_BaseWrapperClf(type):
    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        def deco_reformat_y(func):
            def wrapper(*args, **kwargs):
                y = args[2]
                if type(y) == np.array:
                    args = list(args[:2]) + [Reformat_Ys_MixIn.np_arr_to_index(y)] + list(args[3:])

                ret = func(*args, **kwargs)
                return ret

            return wrapper

        func_names = ['score', 'score_pack', 'fit']
        for func_name in func_names:
            if hasattr(cls, func_name):
                setattr(cls, func_name, deco_reformat_y(getattr(cls, func_name)))


class meta_BaseWrapperClf_with_ABC(meta_BaseWrapperClf, ABCMeta):
    pass


class meta_BaseWrapperReg(type):
    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        # def deco_reformat_y(func):
        #     def wrapper(*args, **kwargs):
        #         y = args[2]
        #         args = list(args[:2]) + [Reformat_Ys_MixIn.np_arr_to_index(y)] + list(args[3:])
        #
        #         ret = func(*args, **kwargs)
        #         return ret
        #
        #     return wrapper
        #
        # func_names = ['score', 'score_pack', 'fit']
        # for func_name in func_names:
        #     if hasattr(cls, func_name):
        #         setattr(cls, func_name, deco_reformat_y(getattr(cls, func_name)))


class meta_BaseWrapperReg_with_ABC(meta_BaseWrapperReg, ABCMeta):
    pass


class Reformat_Ys_MixIn:
    @staticmethod
    def np_arr_to_index(Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_INDEX)

    @staticmethod
    def np_arr_to_onehot(Xs, n=None):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_ONEHOT, n=n)


CLF_METRICS = {
    'accuracy': accuracy_score,
    'confusion_matrix': confusion_matrix,
    'roc_auc_score': roc_auc_score,
    'recall_score': recall_score,
    'precision_score': precision_score,
}


def RMSE(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight, multioutput))


REG_METRIC = {
    r2_score.__name__: r2_score,
    explained_variance_score.__name__: explained_variance_score,
    mean_absolute_error.__name__: mean_absolute_error,
    mean_squared_log_error.__name__: mean_squared_log_error,
    mean_squared_error.__name__: mean_squared_error,
    median_absolute_error.__name__: median_absolute_error,
    RMSE.__name__: RMSE
}


class clf_score_pack_MixIn(Reformat_Ys_MixIn):
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
                pass
                # getattr(self, 'log').info(
                #     f'while "{str(self)}" execute score_pack,'
                #     f' raise "{e}",'
                #     f' skip to applying metric "{key}"\n')
        return ret

    def score_pack(self, X, y):
        y = self.np_arr_to_index(y)
        return self._apply_metric_pack(y, getattr(self, 'predict')(X))


class reg_score_pack_MixIn(Reformat_Ys_MixIn):
    def __init__(self):
        Reformat_Ys_MixIn.__init__(self)
        self._metrics = REG_METRIC

    def _apply_metric(self, Y_true, Y_predict, metric):
        return self._metrics[metric](Y_true, Y_predict)

    def _apply_metric_pack(self, Y_true, Y_predict):
        ret = {}
        for key in self._metrics:
            try:
                ret[key] = self._apply_metric(Y_true, Y_predict, key)
            except BaseException as e:
                getattr(self, 'log').warn(
                    f'while "{str(self)}" execute score_pack,'
                    f' raise "{e}",'
                    f' skip to applying metric "{key}"\n')
        return ret

    def score_pack(self, X, y):
        y = self.np_arr_to_index(y)
        return self._apply_metric_pack(y, getattr(self, 'predict')(X))


class ClfConfidenceMixIn:
    @staticmethod
    def _apply_confidence(proba):
        shape = proba.shape
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


class ClfWrapperMixIn(clf_score_pack_MixIn, PickleMixIn, LoggerMixIn, ClfConfidenceMixIn, etc_MixIn):
    HyperOpt_space = None

    def __init__(self):
        clf_score_pack_MixIn.__init__(self)
        PickleMixIn.__init__(self)
        LoggerMixIn.__init__(self)
        ClfConfidenceMixIn.__init__(self)
        etc_MixIn.__init__(self)


class RegWrapperMixIn(reg_score_pack_MixIn, PickleMixIn, LoggerMixIn, etc_MixIn):
    HyperOpt_space = None

    def __init__(self):
        reg_score_pack_MixIn.__init__(self)
        PickleMixIn.__init__(self)
        LoggerMixIn.__init__(self)
        etc_MixIn.__init__(self)

    def __str__(self):
        return self.__class__.__name__
