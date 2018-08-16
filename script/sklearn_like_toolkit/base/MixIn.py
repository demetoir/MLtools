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


class etc_MixIn:
    @staticmethod
    def _clone(clf):
        return clf.__class__(**clf.get_params())


class yLabelOneHotConvertMixIn:
    @staticmethod
    def np_arr_to_index(x):
        return reformat_np_arr(x, NP_ARRAY_TYPE_INDEX)

    @staticmethod
    def np_arr_to_onehot(x, n=None):
        return reformat_np_arr(x, NP_ARRAY_TYPE_ONEHOT, n=n)


class MetaBaseWrapperClf(type):
    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        def deco_reformat_y(func):
            def wrapper(*args, **kwargs):
                y = args[2]
                if type(y) == np.array:
                    args = list(args[:2]) + [yLabelOneHotConvertMixIn.np_arr_to_index(y)] + list(args[3:])

                ret = func(*args, **kwargs)
                return ret

            return wrapper

        func_names = ['score', 'score_pack', 'fit']
        for func_name in func_names:
            if hasattr(cls, func_name):
                setattr(cls, func_name, deco_reformat_y(getattr(cls, func_name)))


class MetaBaseWrapperClf_with_ABC(MetaBaseWrapperClf, ABCMeta):
    pass


class ClfScorePackMixIn(yLabelOneHotConvertMixIn):
    CLF_METRICS = {
        'accuracy': accuracy_score,
        'confusion_matrix': confusion_matrix,
        'roc_auc_score': roc_auc_score,
        'recall_score': recall_score,
        'precision_score': precision_score,
    }

    def __init__(self):
        yLabelOneHotConvertMixIn.__init__(self)
        self._metrics = self.__class__.CLF_METRICS

    def _apply_metric(self, y_true, y_predict, metric):
        return self._metrics[metric](y_true, y_predict)

    def _apply_metric_pack(self, y_true, y_predict):
        ret = {}
        for key in self._metrics:
            try:
                ret[key] = self._apply_metric(y_true, y_predict, key)
            except BaseException:
                pass
        return ret

    def score_pack(self, x, y):
        y = self.np_arr_to_index(y)
        return self._apply_metric_pack(y, getattr(self, 'predict')(x))


class ClfPredictConfidenceMixIn:
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


class ClfWrapperMixIn(
    ClfScorePackMixIn,
    PickleMixIn,
    LoggerMixIn,
    ClfPredictConfidenceMixIn,
    etc_MixIn
):
    HyperOpt_space = None

    def __init__(self):
        ClfScorePackMixIn.__init__(self)
        PickleMixIn.__init__(self)
        LoggerMixIn.__init__(self)
        ClfPredictConfidenceMixIn.__init__(self)
        etc_MixIn.__init__(self)


class MetaBaseWrapperReg(type):
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


class MetaBaseWrapperReg_with_ABC(MetaBaseWrapperReg, ABCMeta):
    pass


class RegScorePackMixIn(yLabelOneHotConvertMixIn):
    def __init__(self):
        yLabelOneHotConvertMixIn.__init__(self)
        self._metrics = {
            r2_score.__name__: r2_score,
            explained_variance_score.__name__: explained_variance_score,
            mean_absolute_error.__name__: mean_absolute_error,
            mean_squared_log_error.__name__: mean_squared_log_error,
            mean_squared_error.__name__: mean_squared_error,
            median_absolute_error.__name__: median_absolute_error,
            'RMSE': self._RMSE
        }

    @staticmethod
    def _RMSE(y_true, y_predict, sample_weight=None, multioutput='uniform_average'):
        return np.sqrt(mean_squared_error(y_true, y_predict, sample_weight, multioutput))

    def _apply_metric(self, y_true, y_predict, metric):
        return self._metrics[metric](y_true, y_predict)

    def _apply_metric_pack(self, y_true, y_predict):
        ret = {}
        for key in self._metrics:
            try:
                ret[key] = self._apply_metric(y_true, y_predict, key)
            except BaseException as e:
                getattr(self, 'log').warn(
                    f'while "{str(self)}" execute score_pack,'
                    f' raise "{e}",'
                    f' skip to applying metric "{key}"\n')
        return ret

    def score_pack(self, x, y):
        y = self.np_arr_to_index(y)
        return self._apply_metric_pack(y, getattr(self, 'predict')(x))


class RegWrapperMixIn(
    RegScorePackMixIn,
    PickleMixIn,
    LoggerMixIn,
    etc_MixIn
):
    HyperOpt_space = None

    def __init__(self):
        RegScorePackMixIn.__init__(self)
        PickleMixIn.__init__(self)
        LoggerMixIn.__init__(self)
        etc_MixIn.__init__(self)

    def __str__(self):
        return self.__class__.__name__
